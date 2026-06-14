"""Local dataset path resolution for bronze ingest.

Prefers environment variables when they point at real files; otherwise falls back to
the workspace sibling folders (see ``Machine-Learning-Engineering-Group-9-Social-Listener.code-workspace``):

    ../../Yelp_JSON/yelp_dataset/yelp_dataset          # ~9 GB Yelp Open Dataset tar
    ../../TripAdvisor_data_cleaned.csv/TripAdvisor_data_cleaned.csv

Override any default via ``.env`` or shell exports (see ``.env.example``).
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = ROOT.parent.parent

DEFAULT_YELP_TAR = (WORKSPACE_ROOT / "Yelp_JSON" / "yelp_dataset" / "yelp_dataset").resolve()
DEFAULT_TRIPADVISOR_CSV = (
    WORKSPACE_ROOT / "TripAdvisor_data_cleaned.csv" / "TripAdvisor_data_cleaned.csv"
).resolve()


def resolve_data_path(env_name: str, default: Path | None = None) -> Path | None:
    """Return a dataset path from ``env_name`` or ``default`` when the file exists."""
    raw = os.environ.get(env_name)
    if raw:
        path = Path(raw)
        if path.is_file():
            return path
    if default is not None and default.is_file():
        return default
    return None


def yelp_tar_path() -> Path | None:
    return resolve_data_path("YELP_TAR_PATH", DEFAULT_YELP_TAR)


def yelp_reviews_path() -> Path | None:
    return resolve_data_path("YELP_REVIEWS_PATH")


def yelp_business_path() -> Path | None:
    return resolve_data_path("YELP_BUSINESS_PATH")


def tripadvisor_csv_path() -> Path | None:
    return resolve_data_path("TRIPADVISOR_CSV_PATH", DEFAULT_TRIPADVISOR_CSV)
