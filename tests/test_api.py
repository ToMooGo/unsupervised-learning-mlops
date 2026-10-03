"""API tests with small in-memory models and a temporary SQLite database (no MLflow needed)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from unsupervised_mlops.anomaly import GaussianMixtureAnomalyDetector
from unsupervised_mlops.clustering_features import build_kmeans_logreg_pipeline, make_logreg
from unsupervised_mlops.semi_supervised import find_representatives


@pytest.fixture(scope="module")
def bundle(split):
    from app.model_store import ModelBundle

    sup = build_kmeans_logreg_pipeline(n_clusters=20, n_init=1).fit(split.X_train, split.y_train)
    km, _, rep = find_representatives(split.X_train, k=10, n_init=1)
    semi = make_logreg().fit(split.X_train[rep], split.y_train[rep])
    det = GaussianMixtureAnomalyDetector(4, "full", 4, 0.95, 0.25, n_init=1, random_state=0).fit(
        split.X_train
    )
    reps = {
        "train_index": [int(i) for i in rep],
        "pixels": [split.X_train[i].tolist() for i in rep],
        "label_used": [int(v) for v in split.y_train[rep]],
        "label_source": "oracle",
        "split_seed": 42,
    }
    return ModelBundle(
        sup, semi, det, {"supervised": "3", "semi_supervised": "3", "anomaly": "3"}, reps
    )


@pytest.fixture()
def client(bundle, tmp_path):
    from app.main import create_app
    from app.model_store import ModelStore

    store = ModelStore()
    store.set_bundle(bundle)
    app = create_app(
        store=store, database_url=f"sqlite:///{tmp_path / 'test.db'}", load_models=False
    )
    with TestClient(app) as c:
        yield c


@pytest.fixture()
def empty_client(tmp_path):
    from app.main import create_app

    app = create_app(database_url=f"sqlite:///{tmp_path / 'empty.db'}", load_models=False)
    with TestClient(app) as c:
        yield c


def test_health_reports_versions(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["model_versions"]["anomaly"] == "3"


def test_predict_clean_digit(client, split):
    r = client.post("/predict", json={"pixels": split.X_test[0].tolist(), "source": "test"})
    assert r.status_code == 200
    body = r.json()
    assert body["supervised"]["digit"] == int(split.y_test[0])
    assert len(body["supervised"]["probabilities"]) == 10
    assert abs(sum(body["semi_supervised"]["probabilities"]) - 1) < 1e-3
    assert body["anomaly"]["is_anomaly"] is False
    assert body["prediction_id"] == 1


def test_predict_flags_random_noise(client):
    noise = np.random.default_rng(0).integers(0, 17, 64).astype(float).tolist()
    body = client.post("/predict", json={"pixels": noise, "source": "noise"}).json()
    assert body["anomaly"]["is_anomaly"] is True
    assert body["anomaly"]["log_density"] < body["anomaly"]["threshold"]


@pytest.mark.parametrize("pixels", [[0.0] * 63, [0.0] * 63 + [17.0], [-1.0] + [0.0] * 63])
def test_predict_validates_input(client, pixels):
    assert client.post("/predict", json={"pixels": pixels}).status_code == 422


@pytest.mark.parametrize("token", ["NaN", "Infinity"])
def test_predict_rejects_non_finite_pixels(client, token):
    body = '{"pixels": [' + ", ".join([token] + ["0"] * 63) + "]}"
    r = client.post("/predict", content=body, headers={"content-type": "application/json"})
    assert r.status_code == 422


def test_source_label_cannot_carry_markup(client):
    r = client.post(
        "/predict", json={"pixels": [0.0] * 64, "source": "<img src=x onerror=alert(1)>"}
    )
    assert r.status_code == 422


def test_reload_requires_admin_token_when_configured(bundle, tmp_path, monkeypatch):
    from app.main import create_app
    from app.model_store import ModelStore

    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    store = ModelStore()
    store.set_bundle(bundle)
    app = create_app(store=store, database_url=f"sqlite:///{tmp_path / 't.db'}", load_models=False)
    with TestClient(app) as c:
        assert c.post("/reload").status_code == 401
        assert c.post("/reload", headers={"X-Admin-Token": "wrong"}).status_code == 401
        # right token -> passes auth (the reload itself depends on whatever registry is configured)
        assert c.post("/reload", headers={"X-Admin-Token": "s3cret"}).status_code in (200, 503)


def test_samples_endpoint(client):
    s = client.get("/samples?kind=inverted").json()
    assert s["kind"] == "inverted" and len(s["pixels"]) == 64
    assert client.get("/samples?kind=melted").status_code == 422
    a = client.get("/samples?kind=noise&seed=7").json()
    assert a == client.get("/samples?kind=noise&seed=7").json()  # reproducible demos


def test_labeling_roundtrip(client, bundle, split):
    batch = client.get("/labeling/batch").json()
    assert len(batch["items"]) == 10 and batch["n_labelled"] == 0
    idx = batch["items"][0]["train_index"]
    r = client.post(
        "/labeling/labels",
        json={"labels": [{"train_index": idx, "label": int(split.y_train[idx])}]},
    )
    assert r.status_code == 200
    assert r.json()["labelled"] == 1 and r.json()["agreement_with_dataset_labels"] == "1/1"
    assert client.get("/labeling/batch").json()["items"][0]["human_label"] == int(
        split.y_train[idx]
    )
    # indices outside the batch are rejected
    bad = client.post("/labeling/labels", json={"labels": [{"train_index": 99999, "label": 1}]})
    assert bad.status_code == 422


def test_human_labels_feed_the_pipeline(client, tmp_path, split):
    from unsupervised_mlops.db import fetch_human_labels

    idx = client.get("/labeling/batch").json()["items"][1]["train_index"]
    client.post("/labeling/labels", json={"labels": [{"train_index": idx, "label": 7}]})
    labels = fetch_human_labels(f"sqlite:///{tmp_path / 'test.db'}", split_seed=42)
    assert labels[idx] == 7


def test_monitoring_summary_counts_predictions(client, split):
    for i in range(3):
        client.post("/predict", json={"pixels": split.X_test[i].tolist(), "source": "clean"})
    client.post("/predict", json={"pixels": [16.0] * 64, "source": "noise"})
    m = client.get("/monitoring/summary").json()
    assert m["n_predictions"] == 4
    assert m["by_source"]["noise"]["anomalies"] == 1
    assert sum(m["class_counts"].values()) == 4
    assert len(m["recent"]) == 4
    assert m["recent"][0]["source"] == "noise"  # newest first
    assert m["anomaly_rate_pct"] == 25.0


def test_service_without_models_returns_503(empty_client):
    assert empty_client.get("/health").json()["status"] == "no-models"
    assert empty_client.post("/predict", json={"pixels": [0.0] * 64}).status_code == 503


def test_ui_is_served(client):
    r = client.get("/")
    assert r.status_code == 200 and "Digit Lab" in r.text
    assert client.get("/static/app.js").status_code == 200
