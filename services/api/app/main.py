"""FastAPI model service + web UI.

* ``POST /predict``          - digit from both classifiers + anomaly verdict, logged to PostgreSQL
* ``POST /reload``           - hot-swap to the current ``champion`` models in the MLflow registry
* ``GET/POST /labeling/...`` - human-in-the-loop labelling of the 50 representative images
* ``GET /monitoring/summary``- live anomaly rate and prediction mix from the prediction log
* ``GET /``                  - single-page web UI (draw a digit, label, monitor)

Run with ``uvicorn app.main:create_app --factory``.
"""

from __future__ import annotations

import hmac
import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import case, func, select

from unsupervised_mlops.data import CORRUPTIONS, corrupt_digits, load_digits_split
from unsupervised_mlops.db import HumanLabel, Prediction, init_db, make_engine

from .model_store import ModelBundle, ModelStore
from .schemas import AnomalyOutput, ClassifierOutput, LabelsRequest, PredictRequest, PredictResponse

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"), format="%(asctime)s %(levelname)s %(name)s: %(message)s"
)
log = logging.getLogger("api")
STATIC = Path(__file__).parent / "static"
EXPECTED_ANOMALY_PCT = float(os.getenv("EXPECTED_ANOMALY_PCT", "4.0"))


def _classifier_output(model, x: np.ndarray) -> ClassifierOutput:
    proba = model.predict_proba(x)[0]
    full = np.zeros(10)
    full[np.asarray(model.classes_, dtype=int)] = proba
    digit = int(np.argmax(full))
    return ClassifierOutput(
        digit=digit, confidence=float(full[digit]), probabilities=[round(float(p), 5) for p in full]
    )


def _anomaly_output(detector, x: np.ndarray) -> AnomalyOutput:
    log_density = float(detector.score_samples(x)[0])
    threshold = float(detector.threshold_)
    pct = (
        float(detector.density_percentile(x)[0])
        if hasattr(detector, "density_percentile")
        else float("nan")
    )
    is_anomaly = log_density < threshold
    message = (
        "It lies in a low-density region of the training data, so treat the prediction with caution."
        if is_anomaly
        else "Looks like a typical training digit."
    )
    return AnomalyOutput(
        is_anomaly=is_anomaly,
        log_density=round(log_density, 3),
        threshold=round(threshold, 3),
        density_percentile=round(pct, 1),
        message=message,
    )


def create_app(
    store: ModelStore | None = None, database_url: str | None = None, load_models: bool = True
) -> FastAPI:
    store = store or ModelStore(alias=os.getenv("MODEL_ALIAS", "champion"))
    engine = make_engine(database_url or os.getenv("DATABASE_URL", "sqlite:///app.db"))
    Session = init_db(engine)
    state = {"last_poll": time.time()}
    poll_seconds = float(os.getenv("REGISTRY_POLL_SECONDS", "60"))
    split_seed = int(os.getenv("SPLIT_SEED", "42"))
    admin_token = os.getenv("ADMIN_TOKEN", "")
    samples = load_digits_split(random_state=split_seed)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if load_models and store.bundle is None:
            store.try_load()
        yield
        engine.dispose()

    app = FastAPI(
        title="Unsupervised Learning MLOps - digit service",
        version="1.0.0",
        description="K-Means features, semi-supervised learning and GMM anomaly detection on handwritten digits.",
        lifespan=lifespan,
    )
    app.state.store = store
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc: RequestValidationError):
        # Drop the echoed input: it can contain NaN/inf, which is not valid JSON.
        detail = [{k: e[k] for k in ("loc", "msg", "type") if k in e} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"detail": detail})

    def require_models() -> ModelBundle:
        # at most once per poll interval: pick up a new @champion (or the first one) from the registry
        if load_models and poll_seconds > 0 and time.time() - state["last_poll"] > poll_seconds:
            state["last_poll"] = time.time()
            store.refresh_if_stale()
        bundle = store.bundle
        if bundle is None:
            raise HTTPException(
                503, f"No models deployed yet ({store.last_error}). Run the pipeline first."
            )
        return bundle

    @app.get("/", include_in_schema=False)
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/health")
    def health():
        b = store.bundle
        return {
            "status": "ok" if b else "no-models",
            "model_versions": b.versions if b else {},
            "loaded_at": b.loaded_at if b else None,
            "alias": store.alias,
            "detail": None if b else store.last_error,
        }

    @app.post("/reload")
    def reload(x_admin_token: str | None = Header(default=None)):
        """Hot-swap to the registry's current champion. If ADMIN_TOKEN is set, the request must
        carry it in the ``X-Admin-Token`` header (the deploy flow sends it)."""
        if admin_token and not hmac.compare_digest(x_admin_token or "", admin_token):
            raise HTTPException(401, "Missing or invalid X-Admin-Token")
        try:
            b = store.load_from_registry()
        except Exception as exc:
            raise HTTPException(503, f"Reload failed: {exc}") from exc
        return {"status": "reloaded", "model_versions": b.versions, "loaded_at": b.loaded_at}

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest):
        b = require_models()
        t0 = time.perf_counter()
        x = np.asarray(req.pixels, dtype=float).reshape(1, -1)
        sup = _classifier_output(b.supervised, x)
        semi = _classifier_output(b.semi_supervised, x)
        anomaly = _anomaly_output(b.anomaly, x)
        latency = (time.perf_counter() - t0) * 1000
        pred_id = None
        try:
            with Session() as s:
                row = Prediction(
                    source=req.source,
                    pixels=[float(p) for p in req.pixels],
                    supervised_digit=sup.digit,
                    supervised_confidence=sup.confidence,
                    semi_supervised_digit=semi.digit,
                    semi_supervised_confidence=semi.confidence,
                    log_density=anomaly.log_density,
                    density_threshold=anomaly.threshold,
                    is_anomaly=anomaly.is_anomaly,
                    model_versions=b.versions,
                    latency_ms=latency,
                )
                s.add(row)
                s.commit()
                pred_id = row.id
        except Exception as exc:  # serving must not fail because logging failed
            log.error("Could not log prediction: %s", exc)
        return PredictResponse(
            prediction_id=pred_id,
            supervised=sup,
            semi_supervised=semi,
            anomaly=anomaly,
            model_versions=b.versions,
            latency_ms=round(latency, 2),
        )

    @app.get("/samples")
    def sample(
        kind: str = Query("clean", description=f"clean or one of {', '.join(CORRUPTIONS)}"),
        seed: int | None = Query(None, ge=0, description="fix the choice for a reproducible demo"),
    ):
        """A random held-out test digit, optionally corrupted - handy for trying the API."""
        if kind != "clean" and kind not in CORRUPTIONS:
            raise HTTPException(422, f"kind must be 'clean' or one of {CORRUPTIONS}")
        rng = np.random.default_rng(seed)
        i = int(rng.integers(len(samples.X_test)))
        x = samples.X_test[i : i + 1]
        if kind != "clean":
            x = corrupt_digits(x, kind, int(rng.integers(1_000_000)))
        return {
            "kind": kind,
            "pixels": [round(float(v), 3) for v in x[0]],
            "true_label": int(samples.y_test[i]),
        }

    @app.get("/labeling/batch")
    def labeling_batch():
        b = require_models()
        if not b.representatives:
            raise HTTPException(
                404, "The deployed semi-supervised model has no representative images attached."
            )
        reps = b.representatives
        seed = int(reps.get("split_seed", split_seed))
        with Session() as s:
            rows = s.execute(
                select(HumanLabel)
                .where(HumanLabel.split_seed == seed)
                .order_by(HumanLabel.created_at)
            ).scalars()
            existing = {r.train_index: r.label for r in rows}
        items = [
            {"train_index": idx, "pixels": px, "human_label": existing.get(idx)}
            for idx, px in zip(reps["train_index"], reps["pixels"], strict=True)
        ]
        return {
            "split_seed": seed,
            "label_source_of_deployed_model": reps.get("label_source"),
            "items": items,
            "n_labelled": sum(i["human_label"] is not None for i in items),
        }

    @app.post("/labeling/labels")
    def save_labels(req: LabelsRequest):
        b = require_models()
        reps = b.representatives or {}
        allowed = set(reps.get("train_index", []))
        seed = int(reps.get("split_seed", split_seed))
        bad = [item.train_index for item in req.labels if item.train_index not in allowed]
        if bad:
            raise HTTPException(422, f"train_index values not in the labelling batch: {bad[:5]}")
        with Session() as s:
            s.add_all(
                [
                    HumanLabel(
                        split_seed=seed,
                        train_index=i.train_index,
                        label=i.label,
                        labeler=req.labeler,
                    )
                    for i in req.labels
                ]
            )
            s.commit()
            n_done = s.execute(
                select(func.count(func.distinct(HumanLabel.train_index))).where(
                    HumanLabel.split_seed == seed
                )
            ).scalar_one()
        truth = (
            samples.y_train if seed == split_seed else load_digits_split(random_state=seed).y_train
        )
        agree = sum(int(truth[i.train_index]) == i.label for i in req.labels)
        return {
            "saved": len(req.labels),
            "labelled": int(n_done),
            "total": len(allowed),
            "agreement_with_dataset_labels": f"{agree}/{len(req.labels)}",
        }

    @app.get("/monitoring/summary")
    def monitoring_summary(hours: float = Query(168, gt=0, le=24 * 365)):
        """Aggregates are computed in the database; only the 12 most recent rows are fetched."""
        since = datetime.now(UTC) - timedelta(hours=hours)
        window = Prediction.created_at >= since
        anomaly = case((Prediction.is_anomaly, 1), else_=0)
        agree = case((Prediction.supervised_digit == Prediction.semi_supervised_digit, 1), else_=0)
        with Session() as s:
            n, n_anom, mean_latency, n_agree = s.execute(
                select(
                    func.count(),
                    func.sum(anomaly),
                    func.avg(Prediction.latency_ms),
                    func.sum(agree),
                ).where(window)
            ).one()
            class_rows = s.execute(
                select(Prediction.supervised_digit, func.count())
                .where(window)
                .group_by(Prediction.supervised_digit)
            ).all()
            source_rows = s.execute(
                select(Prediction.source, func.count(), func.sum(anomaly))
                .where(window)
                .group_by(Prediction.source)
                .order_by(func.count().desc())
            ).all()
            recent = (
                s.execute(select(Prediction).where(window).order_by(Prediction.id.desc()).limit(12))
                .scalars()
                .all()
            )
        counts = {str(d): 0 for d in range(10)}
        counts.update({str(d): int(c) for d, c in class_rows})
        return {
            "window_hours": hours,
            "n_predictions": int(n),
            "anomaly_rate_pct": round(100 * float(n_anom) / n, 2) if n else None,
            "expected_anomaly_rate_pct": EXPECTED_ANOMALY_PCT,
            "mean_latency_ms": round(float(mean_latency), 2) if n else None,
            "class_counts": counts,
            "models_agree_pct": round(100 * float(n_agree) / n, 1) if n else None,
            "by_source": {
                src: {"n": int(c), "anomalies": int(a or 0)} for src, c, a in source_rows
            },
            "recent": [
                {
                    "id": r.id,
                    "created_at": r.created_at.isoformat(timespec="seconds")
                    if r.created_at
                    else None,
                    "source": r.source,
                    "supervised_digit": r.supervised_digit,
                    "semi_supervised_digit": r.semi_supervised_digit,
                    "is_anomaly": r.is_anomaly,
                    "log_density": round(r.log_density, 1),
                    "pixels": r.pixels,
                }
                for r in recent
            ],
        }

    return app
