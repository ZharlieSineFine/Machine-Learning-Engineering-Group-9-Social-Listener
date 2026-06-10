"""Make the project root importable so tests can `from models.x import y`."""
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# api/ is a top-level dir but its package is `app` (matches the Dockerfile
# CMD `uvicorn app.main:app`). Make `app` importable from tests.
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))


def pytest_collection_modifyitems(config, items):
    """Skip `@pytest.mark.slow` tests unless RUN_SLOW=1 or `-m slow` is set."""
    if config.getoption("-m") and "slow" in config.getoption("-m"):
        return  # explicit opt-in via `-m slow`
    if os.environ.get("RUN_SLOW") == "1":
        return
    skip_slow = pytest.mark.skip(reason="slow test (set RUN_SLOW=1 or `-m slow` to enable)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
