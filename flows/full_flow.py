"""Full flow: train -> evaluate -> deploy, from one config."""

from __future__ import annotations

from prefect import flow, get_run_logger

from flows.deploy_flow import deploy_flow
from flows.eval_flow import eval_flow
from flows.train_flow import train_flow


@flow(name="full-flow", log_prints=True)
def full_flow(config: dict) -> dict:
    trained = train_flow(config)
    evaluated = eval_flow(config, trained["model_versions"])
    deployed = deploy_flow(config, trained["model_versions"], evaluated["passed"])
    summary = {
        "parent_run_id": trained["parent_run_id"],
        "model_versions": trained["model_versions"],
        "gates_passed": evaluated["passed"],
        "deployed": deployed["deployed"],
        "api_reloaded": deployed.get("reloaded", False),
        "test_metrics": evaluated["metrics"],
    }
    get_run_logger().info("Full flow finished: %s", summary)
    return summary


def start(config: dict) -> dict:
    return full_flow(config)
