"""Evaluation flow: re-evaluate the *registered* candidate models on the held-out test set,
apply quality gates and a champion-vs-challenger check. Only a passing candidate is deployed."""

from __future__ import annotations

import mlflow
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE

from unsupervised_mlops import tracking
from unsupervised_mlops.data import load_digits_split
from unsupervised_mlops.gates import all_passed, evaluate_gates, evaluate_models


@task(name="load-models", cache_policy=NONE)
def load_models(versions: dict[str, str]) -> dict:
    return {k: tracking.load_model(tracking.MODEL_NAMES[k], v) for k, v in versions.items()}


@task(name="quality-gates", cache_policy=NONE)
def check_gates(candidate: dict, champion: dict | None, cfg: dict) -> list[dict]:
    return evaluate_gates(
        candidate,
        champion,
        cfg["eval"]["gates"],
        float(cfg["eval"].get("champion_tolerance_pp", 0.5)),
    )


@flow(name="eval-flow", log_prints=True)
def eval_flow(config: dict, model_versions: dict[str, str] | None = None) -> dict:
    logger = get_run_logger()
    tracking.configure(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"])
    if model_versions is None:  # standalone: evaluate the newest registered versions
        model_versions = {k: tracking.latest_version(n) for k, n in tracking.MODEL_NAMES.items()}
        if None in model_versions.values():
            raise RuntimeError("No registered models found - run the train flow first.")
    split = load_digits_split(float(config["data"]["test_size"]), int(config["seed"]))
    candidate = evaluate_models(load_models(model_versions), split, int(config["seed"]))

    champion_versions = {}
    for key, name in tracking.MODEL_NAMES.items():
        mv = tracking.get_alias_version(name, config["deploy"]["alias"])
        if mv is not None:
            champion_versions[key] = str(mv.version)
    champion = None
    if len(champion_versions) == len(tracking.MODEL_NAMES) and champion_versions != model_versions:
        champion = evaluate_models(load_models(champion_versions), split, int(config["seed"]))

    checks = check_gates(candidate, champion, config)
    passed = all_passed(checks)
    with mlflow.start_run(run_name="evaluate", tags={"flow": "eval", "passed": str(passed)}):
        mlflow.log_params({f"candidate_{k}_version": v for k, v in model_versions.items()})
        tracking.log_metrics(candidate, prefix="candidate_")
        if champion:
            tracking.log_metrics(champion, prefix="champion_")
        tracking.log_json(checks, "quality_gates.json")
        mlflow.log_metric("gates_passed", float(passed))
    for c in checks:
        logger.info(
            "%-4s %-48s value=%.4f limit=%.4f",
            "PASS" if c["passed"] else "FAIL",
            c["check"],
            c["value"],
            c["limit"],
        )
    return {
        "passed": passed,
        "checks": checks,
        "model_versions": model_versions,
        "metrics": candidate,
    }


def start(config: dict) -> dict:
    return eval_flow(config)
