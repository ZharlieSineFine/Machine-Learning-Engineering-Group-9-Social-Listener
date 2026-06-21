"""Pydantic schemas — the API <-> dashboard contract.

This file IS the contract referenced in WORKFLOW.md section 6. Any change
here is a PR that updates both the API and the dashboard at once.

Owner: Amelia.
"""
from __future__ import annotations

from pydantic import BaseModel, Field

MAX_BATCH_SIZE = 256


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to classify")


class PredictResponse(BaseModel):
    label: str = Field(..., description="One of: negative, neutral, positive")
    # TODO (member): add per-class probabilities once the model supports
    # predict_proba reliably (LogReg does; LinearSVC does not without
    # CalibratedClassifierCV). Field name suggestion: `probabilities: dict[str, float]`.


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(
        ..., min_length=1, max_length=MAX_BATCH_SIZE,
        description=f"1..{MAX_BATCH_SIZE} review texts to classify",
    )


class BatchPredictResponse(BaseModel):
    labels: list[str] = Field(..., description="One label per input text, in order")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str  # 'pickle' | 'mlflow' | 'none'


class ReloadResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str

class ShadowLogEntry(BaseModel):
    """One row written to the predictions log per /predict call.
    
    TODO (Phase 2): write this to the `predictions` Postgres table
    (see WORKFLOW.md handoff contracts) once Charlie/Ha's schema lands.
    For now, logged in-memory via shadow.py.
    """
    text: str
    production_label: str
    staging_label: str | None  # None if no Staging model is loaded
    stage: str  # 'production' | 'shadow'
    