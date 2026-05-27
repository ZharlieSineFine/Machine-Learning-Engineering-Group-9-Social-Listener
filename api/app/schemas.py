"""Pydantic schemas — the API <-> dashboard contract.

This file IS the contract referenced in WORKFLOW.md section 6. Any change
here is a PR that updates both the API and the dashboard at once.

Owner: Amelia.
"""
from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to classify")


class PredictResponse(BaseModel):
    label: str = Field(..., description="One of: negative, neutral, positive")
    # TODO (member): add per-class probabilities once the model supports
    # predict_proba reliably (LogReg does; LinearSVC does not without
    # CalibratedClassifierCV). Field name suggestion: `probabilities: dict[str, float]`.


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_source: str  # 'pickle' | 'mlflow' | 'none'
