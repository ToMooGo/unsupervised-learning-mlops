"""Monitoring flow: read recent predictions from PostgreSQL and check for input drift.

The GMM threshold is calibrated so that ~4 % of normal digits are flagged. If the live
anomaly rate climbs far above that, the inputs no longer look like the training data.
Run once with ``python run_flow.py --config configs/monitor_flow_config.yaml`` or on a schedule
with ``python -m flows.serve_monitor`` (the ``monitor`` profile in docker-compose.yml).
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import mlflow
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE
from sqlalchemy import select

from unsupervised_mlops import tracking
from unsupervised_mlops.db import Prediction, init_db, make_engine
from unsupervised_mlops.monitoring import summarize_predictions


@task(name="fetch-predictions", cache_policy=NONE)
def fetch_predictions(url: str, window_hours: float) -> list[dict]:
    engine = make_engine(url)
    Session = init_db(engine)
    since = datetime.now(UTC) - timedelta(hours=window_hours)
    with Session() as s:
        rows = s.execute(select(Prediction).where(Prediction.created_at >= since)).scalars().all()
        out = [
            {"digit": r.supervised_digit, "is_anomaly": r.is_anomaly, "source": r.source}
            for r in rows
        ]
    engine.dispose()
    return out


@flow(name="monitor-flow", log_prints=True)
def monitor_flow(config: dict) -> dict:
    logger = get_run_logger()
    mcfg = config["monitor"]
    rows = fetch_predictions(config["database"]["url"], float(mcfg["window_hours"]))
    report = summarize_predictions(rows, mcfg)
    if report["status"] == "insufficient-data":
        logger.info("Only %d predictions in the window - not enough for a verdict.", len(rows))
        return report

    tracking.configure(
        config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"] + "-monitoring"
    )
    with mlflow.start_run(run_name=f"monitor-{datetime.now(UTC):%Y%m%d-%H%M}"):
        mlflow.log_metrics(
            {
                "n_predictions": report["n_predictions"],
                "anomaly_rate_pct": report["anomaly_rate_pct"],
                "top_digit_share_pct": report["top_digit_share_pct"],
            }
        )
        tracking.log_json(report, "monitor_report.json")
    for a in report["alerts"]:
        logger.warning("ALERT: %s", a)
    logger.info(
        "Monitoring status: %s (%d predictions, anomaly rate %.1f%%)",
        report["status"],
        report["n_predictions"],
        report["anomaly_rate_pct"],
    )
    return report


def start(config: dict) -> dict:
    return monitor_flow(config)
