#Pydantic schemas — the API <-> dashboard contract.

from __future__ import annotations

from pydantic import BaseModel, Field

MAX_BATCH_SIZE = 256


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Review text to classify")


class PredictResponse(BaseModel):
    label: str = Field(..., description="One of: negative, neutral, positive")


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
    text: str
    production_label: str
    staging_label: str | None  # None if no Staging model is loaded
    stage: str  # 'production' | 'shadow'
    