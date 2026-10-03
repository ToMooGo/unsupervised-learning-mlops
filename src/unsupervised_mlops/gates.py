"""Deployment quality gates and the champion-vs-challenger rule (pure functions, unit-tested)."""

from __future__ import annotations

import numpy as np


def evaluate_models(models: dict, split, seed: int = 42) -> dict:
    """Test-set metrics for the three models the API serves (used for gates and champion checks)."""
    from .anomaly import evaluate_on_corruptions

    sup, semi, det = models["supervised"], models["semi_supervised"], models["anomaly"]
    table, summary = evaluate_on_corruptions(det, split.X_test, seed)
    return {
        "supervised_accuracy": float(np.mean(sup.predict(split.X_test) == split.y_test)),
        "semi_supervised_accuracy": float(np.mean(semi.predict(split.X_test) == split.y_test)),
        "anomaly_clean_false_positive_pct": summary["clean_false_positive_pct"],
        **{
            f"anomaly_detection_pct_{r.input}": r.flagged_pct
            for r in table.itertuples()
            if r.input != "clean test digits"
        },
    }


def evaluate_gates(
    candidate: dict, champion: dict | None, gates: dict, tolerance_pp: float = 0.5
) -> list[dict]:
    """Return one record per check: ``{"check", "value", "limit", "passed"}``.

    ``candidate`` / ``champion`` hold test-set metrics: ``supervised_accuracy``,
    ``semi_supervised_accuracy``, ``anomaly_clean_false_positive_pct`` and
    ``anomaly_detection_pct_<kind>``. A candidate is deployable only if every check passes.
    """
    m = candidate
    checks = [
        {
            "check": "supervised accuracy",
            "value": m["supervised_accuracy"],
            "limit": gates["supervised_min_accuracy"],
            "passed": m["supervised_accuracy"] >= gates["supervised_min_accuracy"],
        },
        {
            "check": "semi-supervised accuracy",
            "value": m["semi_supervised_accuracy"],
            "limit": gates["semi_supervised_min_accuracy"],
            "passed": m["semi_supervised_accuracy"] >= gates["semi_supervised_min_accuracy"],
        },
        {
            "check": "anomaly false positives on clean digits (%)",
            "value": m["anomaly_clean_false_positive_pct"],
            "limit": gates["anomaly_max_clean_false_positive_pct"],
            "passed": m["anomaly_clean_false_positive_pct"]
            <= gates["anomaly_max_clean_false_positive_pct"],
        },
    ]
    for kind, minimum in gates.get("anomaly_min_detection_pct", {}).items():
        value = m.get(f"anomaly_detection_pct_{kind}", 0.0)
        checks.append(
            {
                "check": f"anomaly detection of '{kind}' inputs (%)",
                "value": value,
                "limit": minimum,
                "passed": value >= minimum,
            }
        )
    if champion:
        tol = tolerance_pp / 100
        for key in ("supervised_accuracy", "semi_supervised_accuracy"):
            checks.append(
                {
                    "check": f"{key} vs champion",
                    "value": m[key],
                    "limit": champion[key] - tol,
                    "passed": m[key] >= champion[key] - tol,
                }
            )
    return checks


def all_passed(checks: list[dict]) -> bool:
    return all(c["passed"] for c in checks)
