"""Single entry point: ``python run_flow.py --config configs/full_flow_config.yaml``.

The config's ``flow`` key picks which flow runs (full | train | eval | deploy | monitor).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)  # relative paths in configs (reports/, sqlite files) resolve from the repo root
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from unsupervised_mlops.config import load_config  # noqa: E402

FLOWS = {
    "full": "flows.full_flow",
    "train": "flows.train_flow",
    "eval": "flows.eval_flow",
    "deploy": "flows.deploy_flow",
    "monitor": "flows.monitor_flow",
}


def main(argv: list[str] | None = None) -> dict:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="YAML config file")
    parser.add_argument("--flow", choices=sorted(FLOWS), help="override the config's `flow` key")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    config["_config_path"] = args.config
    name = args.flow or config.get("flow", "full")
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    result = importlib.import_module(FLOWS[name]).start(config)
    print(json.dumps(result, indent=2, default=str))
    return result


if __name__ == "__main__":
    main()
