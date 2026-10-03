"""Run the monitoring flow on a schedule: ``python -m flows.serve_monitor [--cron "0 * * * *"]``.

Registers a Prefect deployment (visible in the Prefect UI) and keeps serving it. In Docker Compose,
start it with ``docker compose --profile monitor up -d monitor``.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from flows.monitor_flow import monitor_flow  # noqa: E402
from unsupervised_mlops.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/monitor_flow_config.yaml")
    parser.add_argument(
        "--cron", default=os.getenv("MONITOR_CRON", "0 * * * *"), help="default: hourly"
    )
    args = parser.parse_args()
    config = load_config(args.config)
    monitor_flow.serve(name="scheduled-drift-check", cron=args.cron, parameters={"config": config})


if __name__ == "__main__":
    main()
