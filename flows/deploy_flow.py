"""Deploy flow: promote the evaluated models to the ``champion`` alias and hot-reload the API."""

from __future__ import annotations

import os
import time

import httpx
import mlflow
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NONE

from unsupervised_mlops import tracking


@task(name="promote-to-champion", cache_policy=NONE)
def promote(model_versions: dict[str, str], alias: str) -> None:
    for key, version in model_versions.items():
        tracking.set_alias(tracking.MODEL_NAMES[key], version, alias)
        get_run_logger().info("%s v%s -> @%s", tracking.MODEL_NAMES[key], version, alias)


@task(name="reload-api", cache_policy=NONE)
def reload_api(api_url: str, expected: dict[str, str], retries: int, required: bool) -> dict:
    """POST /reload so the running service pulls the new champion, then verify via /health."""
    logger = get_run_logger()
    last_error = None
    retries = retries if required else min(retries, 2)  # don't wait long for an optional API
    for attempt in range(1, retries + 1):
        try:
            headers = (
                {"X-Admin-Token": os.environ["ADMIN_TOKEN"]} if os.getenv("ADMIN_TOKEN") else {}
            )
            with httpx.Client(timeout=60, headers=headers) as client:
                client.post(f"{api_url}/reload").raise_for_status()
                health = client.get(f"{api_url}/health").json()
            served = {k: str(v) for k, v in health.get("model_versions", {}).items()}
            if served == expected:
                logger.info("API at %s now serves %s", api_url, served)
                return {"reloaded": True, "served": served}
            last_error = f"API serves {served}, expected {expected}"
        except httpx.HTTPError as exc:
            last_error = repr(exc)
        logger.warning("Reload attempt %d/%d failed: %s", attempt, retries, last_error)
        time.sleep(min(2 * attempt, 10))
    if required:
        raise RuntimeError(f"Could not reload the API: {last_error}")
    logger.warning(
        "API not reachable - models are promoted; the API will load them on its next start."
    )
    return {"reloaded": False, "error": last_error}


@flow(name="deploy-flow", log_prints=True)
def deploy_flow(
    config: dict, model_versions: dict[str, str] | None = None, passed: bool = True
) -> dict:
    logger = get_run_logger()
    tracking.configure(config["mlflow"]["tracking_uri"], config["mlflow"]["experiment"])
    dep = config["deploy"]
    if model_versions is None:
        model_versions = {k: tracking.latest_version(n) for k, n in tracking.MODEL_NAMES.items()}
    if not passed:
        logger.error("Quality gates failed - keeping the current champion. Nothing deployed.")
        return {"deployed": False, "model_versions": model_versions}
    promote(model_versions, dep["alias"])
    result = reload_api(
        dep["api_url"],
        model_versions,
        int(dep.get("reload_retries", 10)),
        str(dep.get("require_api", False)).lower() == "true",
    )
    with mlflow.start_run(run_name="deploy", tags={"flow": "deploy"}):
        mlflow.log_params({f"{k}_version": v for k, v in model_versions.items()})
        mlflow.log_metric("api_reloaded", float(result["reloaded"]))
    return {"deployed": True, "model_versions": model_versions, **result}


def start(config: dict) -> dict:
    return deploy_flow(config)
