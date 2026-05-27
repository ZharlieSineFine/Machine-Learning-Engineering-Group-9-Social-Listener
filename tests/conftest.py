"""Make the project root importable so tests can `from models.x import y`."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# api/ is a top-level dir but its package is `app` (matches the Dockerfile
# CMD `uvicorn app.main:app`). Make `app` importable from tests.
API_DIR = ROOT / "api"
if str(API_DIR) not in sys.path:
    sys.path.insert(0, str(API_DIR))
