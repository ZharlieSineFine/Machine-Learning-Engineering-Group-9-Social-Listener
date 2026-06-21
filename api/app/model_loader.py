"""Model loading — tries MLflow registry first, falls back to a local pickle.

Order of resolution:
    1. If MLFLOW_TRACKING_URI and MODEL_NAME are set, pull
       `models:/<MODEL_NAME>/<MODEL_STAGE>` from the registry.
    2. Otherwise, load the pickle at MODEL_PICKLE_PATH (defaults to the
       smoke-test artifact written by models/train.py).

Keeping the fallback means the API container can boot in the smoke test
without MLflow being up. Production deployments should always go through
the registry path.

Owner: Amelia.
"""
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
    """Pull `models:/<MODEL_NAME>/<MODEL_STAGE>` from the MLflow registry.

    Returns None (caller falls back to the local pickle) when MLflow isn't
    configured OR the model isn't in the registry yet (nothing promoted) — so the
    API still boots for the demo instead of crash-looping on an empty registry.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MODEL_NAME")
    if not (tracking_uri and model_name):
        return None

    # TODO (member, Phase 2): when DistilBERT (or any non-sklearn model) is
    # registered, switch to `mlflow.pyfunc.load_model` and adapt the call
    # site in main.py to handle DataFrame input. For Phase 1 sklearn, the
    # `mlflow.sklearn` loader returns the raw Pipeline — identical interface
    # to the pickle fallback, so `pipe.predict([text])` works either way.
    import mlflow
    import mlflow.sklearn

    mlflow.set_tracking_uri(tracking_uri)
    stage = os.getenv("MODEL_STAGE", "Production")
    # TODO (member): MLflow 2.9+ deprecated stages in favour of *aliases*.
    # When bumping to MLflow >=3, switch to `models:/sentiment-baseline@production`
    # and have the training DAG set the alias instead of transitioning stages.
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
    """Load the Staging model from MLflow, if one exists.

    Returns None if MLflow isn't configured or no Staging model exists.
    Unlike load_model(), this never falls back to pickle — a missing
    Staging model is normal (not an error).
    """
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