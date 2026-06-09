"""FastAPI service — /health, /predict, /predict/batch, /reload.

Owner: Amelia.
"""
from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException

from app.model_loader import LoadedModel, load_model
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
)

app = FastAPI(title="Sentiment API", version="0.2.0")

# Module-level — mutated by /reload. CPython attribute assignment is atomic,
# which is sufficient for the single-process uvicorn worker the compose
# stack runs. Multi-worker deployments should plumb the reload signal
# through Redis/SIGHUP instead.
_model: LoadedModel | None = load_model()


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        model_source=_model.source if _model else "none",
    )


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest) -> PredictResponse:
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    
    # Input validation
    MAX_CHARS = 1000
    if len(req.text) > MAX_CHARS:
        raise HTTPException(status_code=422, detail=f"text exceeds {MAX_CHARS} character limit")
    
    cleaned = "".join(ch for ch in req.text if ch.isprintable())
    if not cleaned.strip():
        raise HTTPException(status_code=422, detail="text is empty after cleaning")

<<<<<<< HEAD
=======
    # TODO (member): input validation/sanitisation
    #   - length cap (Pydantic min_length only; add max_length=10_000)
    #   - strip control chars
>>>>>>> feature/full_flow
    pred = _model.pipeline.predict([req.text])
    return PredictResponse(label=str(pred[0]))


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Batch inference. Cap at MAX_BATCH_SIZE (set in schemas)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    preds = _model.pipeline.predict(req.texts)
    return BatchPredictResponse(labels=[str(p) for p in preds])


@app.post("/reload", response_model=ReloadResponse)
def reload_model(x_admin_token: str | None = Header(default=None)) -> ReloadResponse:
    """Admin: re-pull the model from MLflow without restarting the container.

    Guarded by an admin token (env var ADMIN_TOKEN). The token must be
    sent in the `X-Admin-Token` header. If ADMIN_TOKEN is unset the
    endpoint is disabled — we won't ship a no-auth reload to production
    by accident.
    """
    expected = os.environ.get("ADMIN_TOKEN")
    if not expected:
        raise HTTPException(status_code=503, detail="reload endpoint disabled (ADMIN_TOKEN unset)")
    if not x_admin_token or x_admin_token != expected:
        raise HTTPException(status_code=401, detail="invalid or missing admin token")

    global _model
    _model = load_model()
    return ReloadResponse(
        status="ok",
        model_loaded=_model is not None,
        model_source=_model.source if _model else "none",
    )
