"""The deployment gates and the drift check are the system's safety features - test them directly."""

import pytest

from unsupervised_mlops.config import load_config
from unsupervised_mlops.gates import all_passed, evaluate_gates
from unsupervised_mlops.monitoring import summarize_predictions

GATES = {
    "supervised_min_accuracy": 0.95,
    "semi_supervised_min_accuracy": 0.85,
    "anomaly_max_clean_false_positive_pct": 8.0,
    "anomaly_min_detection_pct": {"noise": 95, "inverted": 95},
}
GOOD = {
    "supervised_accuracy": 0.976,
    "semi_supervised_accuracy": 0.924,
    "anomaly_clean_false_positive_pct": 2.9,
    "anomaly_detection_pct_noise": 100.0,
    "anomaly_detection_pct_inverted": 100.0,
}


def test_good_candidate_without_champion_passes():
    checks = evaluate_gates(GOOD, None, GATES)
    assert all_passed(checks)
    assert len(checks) == 5  # 3 fixed gates + 2 detection gates, no champion checks


@pytest.mark.parametrize(
    "override",
    [
        {"supervised_accuracy": 0.94},
        {"semi_supervised_accuracy": 0.80},
        {"anomaly_clean_false_positive_pct": 9.5},
        {"anomaly_detection_pct_noise": 90.0},
    ],
)
def test_each_absolute_gate_can_fail(override):
    checks = evaluate_gates({**GOOD, **override}, None, GATES)
    assert not all_passed(checks)


def test_champion_blocks_a_regression_from_one_wrong_label():
    """The observed human-labelling incident: one wrong representative label dropped the
    semi-supervised model from 92.4% to 88.9% - above the 85% floor, but worse than the champion."""
    challenger = {**GOOD, "semi_supervised_accuracy": 0.889}
    checks = evaluate_gates(challenger, GOOD, GATES, tolerance_pp=0.5)
    failed = [c["check"] for c in checks if not c["passed"]]
    assert failed == ["semi_supervised_accuracy vs champion"]


def test_champion_tolerance_allows_noise_level_differences():
    challenger = {**GOOD, "supervised_accuracy": GOOD["supervised_accuracy"] - 0.004}
    assert all_passed(evaluate_gates(challenger, GOOD, GATES, tolerance_pp=0.5))


MON = {"min_predictions": 20, "max_anomaly_rate_pct": 12.0, "max_class_share_pct": 40.0}


def _rows(n, n_anom=0, digit_of=lambda i: i % 10):
    return [{"digit": digit_of(i), "is_anomaly": i < n_anom} for i in range(n)]


def test_monitor_needs_enough_traffic():
    assert summarize_predictions(_rows(5), MON)["status"] == "insufficient-data"


def test_monitor_ok_on_normal_traffic():
    rep = summarize_predictions(_rows(100, n_anom=4), MON)
    assert rep["status"] == "ok" and rep["anomaly_rate_pct"] == 4.0


def test_monitor_alerts_on_anomaly_rate_and_class_skew():
    rep = summarize_predictions(_rows(100, n_anom=20), MON)
    assert rep["status"] == "alert" and "anomaly rate" in rep["alerts"][0]
    skewed = summarize_predictions(_rows(100, digit_of=lambda i: 7 if i < 60 else i % 10), MON)
    assert any("digit 7" in a for a in skewed["alerts"])


def test_repo_config_gates_match_the_tests():
    from pathlib import Path

    cfg = load_config(Path(__file__).resolve().parents[1] / "configs" / "full_flow_config.yaml")
    assert cfg["eval"]["gates"]["supervised_min_accuracy"] == GATES["supervised_min_accuracy"]
    assert cfg["eval"]["champion_tolerance_pp"] == 0.5
