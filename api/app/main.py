#FastAPI service — /health, /predict, /predict/batch, /reload.

from __future__ import annotations

import os

from fastapi import FastAPI, Header, HTTPException

from app.model_loader import LoadedModel, load_model, load_staging_model
from app.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    HealthResponse,
    PredictRequest,
    PredictResponse,
    ReloadResponse,
    ShadowLogEntry,
)
from app import shadow

app = FastAPI(title="Sentiment API", version="0.2.0")

_model: LoadedModel | None = load_model()

# Staging model — optional. Present only when Van has promoted a candidate
# to Staging in MLflow. Shadow predictions run silently alongside Production.
_staging_model: LoadedModel | None = load_staging_model()

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

    # Production prediction — this is what the caller receives.
    production_label = str(_model.pipeline.predict([req.text])[0])

    # Shadow prediction — runs silently if a Staging model is loaded.
    staging_label: str | None = None
    if _staging_model is not None:
        staging_label = str(_staging_model.pipeline.predict([req.text])[0])

    # Log both to shadow store (dashboard A/B tile reads from here).
    shadow.record(
        text=req.text,
        production_label=production_label,
        staging_label=staging_label,
    )

    return PredictResponse(label=production_label)


@app.post("/predict/batch", response_model=BatchPredictResponse)
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Batch inference. Cap at MAX_BATCH_SIZE (set in schemas)."""
    if _model is None:
        raise HTTPException(status_code=503, detail="model not loaded")

    production_labels = [str(p) for p in _model.pipeline.predict(req.texts)]

    # Shadow the batch too if Staging is loaded.
    if _staging_model is not None:
        staging_labels = [str(p) for p in _staging_model.pipeline.predict(req.texts)]
        for text, prod, stag in zip(req.texts, production_labels, staging_labels):
            shadow.record(text=text, production_label=prod, staging_label=stag)
    else:
        for text, prod in zip(req.texts, production_labels):
            shadow.record(text=text, production_label=prod, staging_label=None)

    return BatchPredictResponse(labels=production_labels)


@app.get("/shadow/log", response_model=list[ShadowLogEntry])
def shadow_log() -> list[ShadowLogEntry]:
    """Return all shadow prediction pairs. Used by the dashboard A/B tile.
    
    TODO (Phase 2): replace with a DB query once predictions table is live.
    """
    return shadow.get_log()


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

    global _model, _staging_model
    _model = load_model()
    _staging_model = load_staging_model()
    return ReloadResponse(
        status="ok",
        model_loaded=_model is not None,
        model_source=_model.source if _model else "none",
    )
