"""Train flow: Parts A, B and C -> MLflow (params, metrics, figures, models) -> Model Registry.

Every part runs as a Prefect task and logs to its own nested MLflow run under one parent run.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import mlflow
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE

from unsupervised_mlops import reporting, tracking
from unsupervised_mlops.anomaly import error_rate_by_flag, run_part_c
from unsupervised_mlops.clustering_features import run_part_a
from unsupervised_mlops.data import DigitsSplit, load_digits_split
from unsupervised_mlops.semi_supervised import run_part_b

ROOT = Path(__file__).resolve().parents[1]


@task(name="load-digits", cache_policy=NONE)
def load_data(cfg: dict) -> DigitsSplit:
    split = load_digits_split(
        test_size=float(cfg["data"]["test_size"]), random_state=int(cfg["seed"])
    )
    get_run_logger().info("Digits split: %s", split.sizes)
    return split


@task(name="part-a-clustering-features", cache_policy=NONE)
def part_a(split: DigitsSplit, cfg: dict, parent_run_id: str):
    with mlflow.start_run(
        run_name="A-clustering-features", nested=True, parent_run_id=parent_run_id
    ):
        res = run_part_a(split, cfg["part_a"], seed=int(cfg["seed"]))
        tracking.log_params(res.params)
        tracking.log_metrics(res.metrics)
        tracking.log_table(res.grid_scores, "tables/k_grid_search.csv")
        tracking.log_table(res.diagnostics, "tables/inertia_silhouette.csv")
        if res.robustness is not None:
            tracking.log_table(res.robustness, "tables/robustness.csv")
        if res.ablation is not None:
            tracking.log_table(res.ablation, "tables/ablation_scaling.csv")
        res.run_id = mlflow.active_run().info.run_id
    get_run_logger().info(
        "Part A: baseline %.4f -> tuned %.4f (k=%d)",
        res.metrics["baseline_accuracy"],
        res.metrics["tuned_accuracy"],
        res.metrics["best_k"],
    )
    return res


@task(name="fetch-human-labels", cache_policy=NONE, retries=2, retry_delay_seconds=5)
def get_human_labels(cfg: dict) -> dict[int, int] | None:
    if str(cfg["part_b"].get("label_source", "oracle")).lower() != "human":
        return None
    from unsupervised_mlops.db import fetch_human_labels

    labels = fetch_human_labels(cfg["database"]["url"], int(cfg["seed"]))
    get_run_logger().info("Fetched %d human labels from the database", len(labels))
    return labels


@task(name="part-b-semi-supervised", cache_policy=NONE)
def part_b(split: DigitsSplit, cfg: dict, parent_run_id: str, human_labels):
    with mlflow.start_run(run_name="B-semi-supervised", nested=True, parent_run_id=parent_run_id):
        res = run_part_b(split, cfg["part_b"], seed=int(cfg["seed"]), human_labels=human_labels)
        tracking.log_params(res.params)
        tracking.log_metrics(res.metrics)
        tracking.log_table(res.percentile_sweep, "tables/percentile_sweep.csv")
        if res.active_learning is not None:
            tracking.log_table(res.active_learning, "tables/active_learning.csv")
            tracking.log_table(res.random_curve, "tables/random_label_curve.csv")
        if res.robustness is not None:
            tracking.log_table(res.robustness, "tables/robustness.csv")
        # the labelling page of the web UI reads these images back from the champion's run
        tracking.log_json(res.representatives, "labeling/representatives.json")
        res.run_id = mlflow.active_run().info.run_id
    get_run_logger().info(
        "Part B: random-50 %.4f -> partial propagation %.4f",
        res.metrics["random50_accuracy"],
        res.metrics["partial_propagation_accuracy"],
    )
    return res


@task(name="part-c-anomaly-detection", cache_policy=NONE)
def part_c(split: DigitsSplit, cfg: dict, parent_run_id: str):
    with mlflow.start_run(run_name="C-anomaly-detection", nested=True, parent_run_id=parent_run_id):
        res = run_part_c(split, cfg["part_c"], seed=int(cfg["seed"]))
        tracking.log_params(res.params)
        tracking.log_metrics(res.metrics)
        tracking.log_table(res.benchmark, "tables/benchmark_all_seeds.csv")
        tracking.log_table(res.benchmark_summary, "tables/benchmark_summary.csv")
        tracking.log_table(res.digits_ic_table, "tables/digits_bic_aic.csv")
        tracking.log_table(res.corruption_table, "tables/digits_corruptions.csv")
        res.run_id = mlflow.active_run().info.run_id
    get_run_logger().info(
        "Part C: digits clean FP %.1f%%, ROC-AUC vs corruptions %.3f",
        res.metrics["digits_clean_false_positive_pct"],
        res.metrics["digits_roc_auc_clean_vs_corrupted"],
    )
    return res


@task(name="register-models", cache_policy=NONE)
def register_models(a, b, c, split: DigitsSplit) -> dict[str, str]:
    example = split.X_test[:2]
    versions = {}
    for key, res, extra in [
        (
            "supervised",
            a,
            {"test_accuracy": a.metrics["tuned_accuracy"], "best_k": a.metrics["best_k"]},
        ),
        (
            "semi_supervised",
            b,
            {
                "test_accuracy": b.metrics["partial_propagation_accuracy"],
                "n_human_labels": b.metrics["n_human_labels"],
            },
        ),
        (
            "anomaly",
            c,
            {"threshold": c.digits_detector.threshold_, "contamination": c.params["contamination"]},
        ),
    ]:
        model = c.digits_detector if key == "anomaly" else res.model
        with mlflow.start_run(run_id=res.run_id, nested=True):
            versions[key] = tracking.log_model(model, tracking.MODEL_NAMES[key], example, extra)
    get_run_logger().info("Registered model versions: %s", versions)
    return versions


@task(name="write-reports", cache_policy=NONE)
def write_reports(a, b, c, split, cfg: dict, extra: dict, run_info: dict) -> dict:
    fig_dir = ROOT / cfg["reports"]["figures_dir"]
    figs = reporting.make_figures(a, b, c, split, fig_dir)
    for p in figs.values():
        mlflow.log_artifact(str(p), "figures")
    rep_dir = ROOT / cfg["reports"]["dir"]
    reporting.write_metrics_json(rep_dir / "metrics.json", a, b, c, extra, run_info)
    reporting.write_results_md(rep_dir / "RESULTS.md", a, b, c, extra, run_info)
    mlflow.log_artifact(str(rep_dir / "metrics.json"))
    mlflow.log_artifact(str(rep_dir / "RESULTS.md"))
    return {k: str(v) for k, v in figs.items()}


@flow(name="train-flow", log_prints=True)
def train_flow(config: dict) -> dict:
    tracking.configure(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"])
    split = load_data(config)
    with mlflow.start_run(run_name="train", tags={"flow": "train"}) as parent:
        pid = parent.info.run_id
        with tempfile.TemporaryDirectory() as tmp:
            cfg_path = Path(tmp) / "config.json"
            cfg_path.write_text(json.dumps(config, indent=2))
            mlflow.log_artifact(str(cfg_path))
        mlflow.log_params({"seed": config["seed"], **split.sizes})
        a = part_a(split, config, pid)
        b = part_b(split, config, pid, get_human_labels(config))
        c = part_c(split, config, pid)
        extra = error_rate_by_flag(a.model, c.digits_detector, split.X_test, split.y_test)
        tracking.log_metrics(extra, prefix="digits_")
        versions = register_models(a, b, c, split)
        run_info = {
            "parent_run_id": pid,
            "config": config.get("_config_path", "n/a"),
            "model_versions": versions,
        }
        write_reports(a, b, c, split, config, extra, run_info)
        mlflow.log_metrics(
            {
                "A_tuned_accuracy": a.metrics["tuned_accuracy"],
                "B_partial_propagation_accuracy": b.metrics["partial_propagation_accuracy"],
                "C_digits_clean_false_positive_pct": c.metrics["digits_clean_false_positive_pct"],
            }
        )
    return {"parent_run_id": pid, "model_versions": versions}


def start(config: dict) -> dict:
    return train_flow(config)
