import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for p in (ROOT / "src", ROOT / "services" / "api", ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
os.environ.setdefault("PREFECT_SERVER_ANALYTICS_ENABLED", "false")

from unsupervised_mlops.data import load_digits_split  # noqa: E402


@pytest.fixture(scope="session")
def split():
    return load_digits_split(random_state=42)
