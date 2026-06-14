"""FastAPI service — /health, /predict, /predict/batch, /reload.

Owner: Amelia.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException

from app.model_loader import load_model_set
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
)
from app.shadow import predict_one_with_shadow, predict_with_shadow
from models.inference import ModelSet

app = FastAPI(title="Sentiment API", version="0.3.0")

# Module-level — mutated by /reload. CPython attribute assignment is atomic,
# which is sufficient for the single-process uvicorn worker the compose
# stack runs. Multi-worker deployments should plumb the reload signal
# through Redis/SIGHUP instead.
_models: ModelSet = load_model_set()


def _validate_text(text: str) -> str:
    max_chars = 1000
    if len(text) > max_chars:
        raise HTTPException(
            status_code=422,
            detail=f"text exceeds {max_chars} character limit",
        )
    cleaned = "".join(ch for ch in text if ch.isprintable())
    if not cleaned.strip():
        raise HTTPException(status_code=422, detail="text is empty after cleaning")
    return text


def _validate_texts(texts: list[str]) -> list[str]:
    return [_validate_text(t) for t in texts]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    production = _models.production
    shadow = _models.shadow
    return HealthResponse(
        status="ok",
        model_loaded=production is not None,
        model_source=production.source if production else "none",
        shadow_model_loaded=shadow is not None,
        shadow_model_source=shadow.source if shadow else "none",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _models.production is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    text = _validate_text(req.text)
    label = predict_one_with_shadow(_models, text, review_id=req.review_id)
    return PredictResponse(label=label)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Batch inference. Cap at MAX_BATCH_SIZE (set in schemas)."""
    if _models.production is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    texts = _validate_texts(req.texts)
    if req.review_ids is not None and len(req.review_ids) != len(texts):
        raise HTTPException(
            status_code=422,
            detail="review_ids must have the same length as texts",
        )
    labels = predict_with_shadow(
        _models,
        texts,
        review_ids=req.review_ids,
    )
    return BatchPredictResponse(labels=labels)


@app.post("/reload", response_model=ReloadResponse)
def reload_model(x_admin_token: str | None = Header(default=None)) -> ReloadResponse:
    """Admin: re-pull models from MLflow without restarting the container."""
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="reload endpoint disabled (ADMIN_TOKEN unset)",
        )
    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="invalid or missing admin token")

    global _models
    _models = load_model_set()
    production = _models.production
    shadow = _models.shadow
    return ReloadResponse(
        status="ok",
        model_loaded=production is not None,
        model_source=production.source if production else "none",
        shadow_model_loaded=shadow is not None,
        shadow_model_source=shadow.source if shadow else "none",
    )
