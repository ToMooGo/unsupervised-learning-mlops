"""Drift checks on the prediction log (pure function, unit-tested).

The GMM threshold is calibrated so that about 4 % of normal digits are flagged. A live anomaly
rate far above that means the inputs no longer look like the training data.
"""

from __future__ import annotations

from collections import Counter


def summarize_predictions(rows: list[dict], cfg: dict) -> dict:
    """``rows``: dicts with ``digit`` and ``is_anomaly``. ``cfg``: the ``monitor`` config section."""
    n = len(rows)
    report: dict = {"n_predictions": n, "alerts": []}
    if n < int(cfg["min_predictions"]):
        report["status"] = "insufficient-data"
        return report
    anomaly_rate = 100 * sum(bool(r["is_anomaly"]) for r in rows) / n
    counts = Counter(int(r["digit"]) for r in rows)
    top_digit, top_count = counts.most_common(1)[0]
    top_share = 100 * top_count / n
    report.update(
        {
            "anomaly_rate_pct": anomaly_rate,
            "top_digit": top_digit,
            "top_digit_share_pct": top_share,
            "class_counts": {str(k): v for k, v in sorted(counts.items())},
        }
    )
    if anomaly_rate > float(cfg["max_anomaly_rate_pct"]):
        report["alerts"].append(
            f"anomaly rate {anomaly_rate:.1f}% > {cfg['max_anomaly_rate_pct']}%"
        )
    if top_share > float(cfg["max_class_share_pct"]):
        report["alerts"].append(f"digit {top_digit} is {top_share:.1f}% of predictions")
    report["status"] = "alert" if report["alerts"] else "ok"
    return report
