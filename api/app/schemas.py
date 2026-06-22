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
    review_id: int | None = Field(
        default=None,
        description="Optional Postgres reviews.id for prediction logging",
    )


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
    review_ids: list[int] | None = Field(
        default=None,
        description="Optional review ids aligned with texts for prediction logging",
    )


class BatchPredictResponse(BaseModel):
    labels: list[str] = Field(..., description="One label per input text, in order")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str  # 'pickle' | 'mlflow' | 'none'
    shadow_model_loaded: bool = False
    shadow_model_source: str = "none"


class ReloadResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str
    shadow_model_loaded: bool = False
    shadow_model_source: str = "none"
