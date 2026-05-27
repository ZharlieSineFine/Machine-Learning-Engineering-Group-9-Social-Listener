# API — FastAPI Serving Layer

**Owner:** Amelia

A REST service that loads the `Production` model from the MLflow registry and exposes `/predict`.

## Endpoints (target)

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/predict` | Single or batch sentiment prediction |
| `POST` | `/predict/batch` | (Phase 2) Vectorised batch prediction |
| `POST` | `/admin/reload` | Re-pull current `Production` model without restart |
| `GET` | `/health` | Liveness probe (used by Compose + CI) |
| `GET` | `/info` | Loaded model name + version + git SHA |

## Contract (Pydantic schemas live in `app/schemas.py`)

```jsonc
// Request
{ "texts": ["This phone is amazing", "Worst battery ever"] }

// Response
{
  "predictions": [
    { "text": "This phone is amazing", "label": "positive", "score": 0.97 },
    { "text": "Worst battery ever",    "label": "negative", "score": 0.94 }
  ],
  "model": { "name": "sentiment-distilbert", "version": 7 }
}
```

Downstream (dashboard) depends on this shape — bump the version field, don't break it silently.

## Local
```
docker compose up api
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
     -d '{"texts":["pretty good"]}'
```

## Phase 1 stub
Return a hardcoded `{label: "positive", score: 0.5}` so the dashboard can be wired up before the registry is live.
