"""Promote a freshly trained MLflow model version to the Production stage.

The FastAPI service (``api/app/model_loader.py``) only serves
``models:/{MODEL_NAME}/Production``. ``models.train.run`` registers a new
version but leaves it unstaged, so without this step a new model never reaches
the API. This closes the loop inside the cycle DAG, gated on a metric so a bad
run doesn't ship.

Owner: Van (Modeler).

NOTE: ported into the ``data_loader`` branch during the Airflow integration so
``medallion_train_cycle`` runs end-to-end. Coordinate the eventual merge with
Van, who owns ``models/``.
"""
from __future__ import annotations

import os
from typing import Optional

DEFAULT_MODEL_NAME = "sentiment-baseline"
DEFAULT_MIN_F1_MACRO = 0.5


def promote_to_production(
    version: Optional[str],
    metrics: dict,
    *,
    model_name: Optional[str] = None,
    min_f1_macro: float = DEFAULT_MIN_F1_MACRO,
) -> bool:
    """Transition ``version`` to Production if it clears the metric gate.

    Returns True if promoted, False if skipped (gate failed, no version, or
    MLflow not configured). Never raises on a skip — only a real MLflow error
    propagates.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        print("[promote] MLFLOW_TRACKING_URI unset — skipping promotion")
        return False
    if not version:
        print("[promote] no model version to promote — skipping")
        return False

    f1_macro = metrics.get("f1_macro")
    if f1_macro is None or f1_macro < min_f1_macro:
        print(
            f"[promote] gate failed: f1_macro={f1_macro} < {min_f1_macro} — "
            f"leaving version {version} unstaged"
        )
        return False

    from mlflow.tracking import MlflowClient

    name = model_name or os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=name,
        version=str(version),
        stage="Production",
        archive_existing_versions=True,
    )
    print(
        f"[promote] {name} v{version} -> Production "
        f"(f1_macro={f1_macro:.3f} >= {min_f1_macro})"
    )
    return True
