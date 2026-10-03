"""End-to-end: run the quick config through Prefect + MLflow (local SQLite registry)."""

import pytest

pytestmark = pytest.mark.integration


def test_full_flow_trains_evaluates_and_promotes(tmp_path, monkeypatch):
    from pathlib import Path

    from unsupervised_mlops import tracking
    from unsupervised_mlops.config import load_config

    root = Path(__file__).resolve().parents[1]
    cfg = load_config(
        root / "configs" / "quick_flow_config.yaml",
        overrides={
            "mlflow": {"tracking_uri": f"sqlite:///{tmp_path / 'mlflow.db'}"},
            "reports": {
                "dir": str(tmp_path / "reports"),
                "figures_dir": str(tmp_path / "reports" / "figures"),
            },
            "deploy": {"api_url": "http://127.0.0.1:9", "require_api": False, "reload_retries": 1},
        },
    )
    monkeypatch.chdir(tmp_path)  # mlruns/ artefacts land in the temp dir

    from prefect.testing.utilities import prefect_test_harness

    from flows.full_flow import full_flow

    with prefect_test_harness():  # isolated, temporary Prefect API
        result = full_flow(cfg)
    assert result["gates_passed"] is True
    assert result["deployed"] is True
    assert result["test_metrics"]["supervised_accuracy"] > 0.95
    assert result["test_metrics"]["semi_supervised_accuracy"] > 0.85
    for name in tracking.MODEL_NAMES.values():
        assert tracking.get_alias_version(name, "champion") is not None
    assert (tmp_path / "reports" / "RESULTS.md").exists()
    assert len(list((tmp_path / "reports" / "figures").glob("*.png"))) >= 8
