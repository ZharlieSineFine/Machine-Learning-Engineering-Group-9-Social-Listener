#Model loading — tries MLflow registry first, falls back to a local pickle.

from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

DEFAULT_PICKLE = Path(os.getenv(
    "MODEL_PICKLE_PATH",
    str(Path(__file__).resolve().parents[2] / "models" / "artifacts" / "baseline.pkl"),
))


@dataclass
class LoadedModel:
    pipeline: Any
    source: str  # 'mlflow' | 'pickle'


def _try_mlflow() -> Optional[LoadedModel]:

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MODEL_NAME")
    if not (tracking_uri and model_name):
        return None

    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(tracking_uri)
    stage = os.getenv("MODEL_STAGE", "Production")

    uri = f"models:/{model_name}/{stage}"
    try:
        pipe = mlflow.sklearn.load_model(uri)
    except Exception as exc:
        print(f"[model_loader] MLflow load failed for {uri} ({exc}); falling back to pickle")
        return None
    return LoadedModel(pipeline=pipe, source="mlflow")


def _load_pickle(path: Path) -> LoadedModel:
    with open(path, "rb") as f:
        pipe = pickle.load(f)
    return LoadedModel(pipeline=pipe, source="pickle")


def load_model() -> Optional[LoadedModel]:
    """Best-effort load. Returns None if both paths fail (API still boots)."""
    via_mlflow = _try_mlflow()
    if via_mlflow is not None:
        return via_mlflow
    if DEFAULT_PICKLE.exists():
        return _load_pickle(DEFAULT_PICKLE)
    print(f"[model_loader] No model available at {DEFAULT_PICKLE} and MLflow not configured")
    return None

def load_staging_model() -> Optional[LoadedModel]:

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MODEL_NAME")
    if not (tracking_uri and model_name):
        return None

    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(tracking_uri)
    try:
        uri = f"models:/{model_name}/Staging"
        pipe = mlflow.sklearn.load_model(uri)
        return LoadedModel(pipeline=pipe, source="mlflow-staging")
    except Exception:
        # No Staging model promoted yet — this is expected during Phase 2 ramp-up.
        return None