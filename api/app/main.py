"""FastAPI service — /health and /predict.

Phase 1 surface area (per WORKFLOW.md):
    GET  /health   — liveness + which model is loaded
    POST /predict  — single-text sentiment classification

Phase 2 (member work, see TODOs below):
    POST /predict supports batch input
    POST /reload  — re-pull model from MLflow without container restart

Owner: Amelia.
"""
from __future__ import annotations

from fastapi import FastAPI, HTTPException

from app.model_loader import LoadedModel, load_model
from app.schemas import HealthResponse, PredictRequest, PredictResponse

app = FastAPI(title="Sentiment API", version="0.1.0")

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

    # TODO (member): add input validation/sanitisation here.
    #   - reject text > N chars (DoS guard)
    #   - strip control chars
    #   - log-and-return for non-string input edge cases
    pred = _model.pipeline.predict([req.text])
    label = str(pred[0])
    return PredictResponse(label=label)


# TODO (member, Phase 2): /predict batch endpoint
#   @app.post("/predict/batch")
#   def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse: ...
#
# TODO (member, Phase 2): /reload admin endpoint
#   Re-runs load_model() so a freshly-promoted model is picked up without
#   restarting the container. Guard with an admin token from .env.
