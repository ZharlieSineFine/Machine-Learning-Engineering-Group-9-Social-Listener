# API — FastAPI Serving Layer · Group 9

**Owner:** Amelia  
**Stack:** FastAPI · Pydantic · scikit-learn · MLflow  
**Entry point:** `api/app/main.py`  
**Runs at:** `http://localhost:8000`

---

## What this folder contains

```
api/
├── app/
│   ├── main.py          # FastAPI app — all route handlers
│   ├── model_loader.py  # Loads model from MLflow registry or local pickle
│   ├── schemas.py       # Pydantic request/response schemas (API contract)
│   ├── shadow.py        # In-memory shadow deploy logger
│   └── __init__.py
├── requirements.txt
└── README.md
```

---

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness probe — returns model load status and source |
| `POST` | `/predict` | Single-text sentiment prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 256 texts) |
| `GET` | `/shadow/log` | Returns all shadow prediction pairs (read by the MLOps dashboard) |
| `POST` | `/reload` | Re-pull the Production model from MLflow without restarting the container |

---

## Request / response shapes

### `POST /predict`
```jsonc
// Request
{ "text": "The coffee was cold and the staff were rude." }

// Response
{ "label": "negative" }
// label is one of: "positive", "neutral", "negative"
```

### `POST /predict/batch`
```jsonc
// Request — up to 256 texts
{ "texts": ["Great latte!", "Waited 30 minutes.", "It was fine."] }

// Response — one label per input, in order
{ "labels": ["positive", "negative", "neutral"] }
```

### `GET /health`
```jsonc
{
  "status": "ok",
  "model_loaded": true,
  "model_source": "mlflow"   // "mlflow" | "pickle" | "none"
}
```

### `GET /shadow/log`
```jsonc
[
  {
    "text": "Loved the flat white.",
    "production_label": "positive",
    "staging_label": "positive",   // null if no Staging model is loaded
    "stage": "shadow"              // "shadow" | "production"
  }
]
```

### `POST /reload`
Requires `X-Admin-Token: <value>` header matching the `ADMIN_TOKEN` env var.
```jsonc
{ "status": "ok", "model_loaded": true, "model_source": "mlflow" }
```

---

## How model loading works (`model_loader.py`)

On startup, `load_model()` tries sources in order:

```
1. MLflow registry  → if MLFLOW_TRACKING_URI and MODEL_NAME are both set,
                       pulls models:/<MODEL_NAME>/<MODEL_STAGE> (default: Production)
2. Local pickle     → models/artifacts/baseline.pkl
                       (override with MODEL_PICKLE_PATH env var)
3. None             → API still boots, /predict returns 503
```

`load_staging_model()` runs the same MLflow path but targets the `Staging` stage. A missing Staging model is normal and not an error — shadow predictions simply don't run until Van promotes one.

---

## How shadow deploy works (`shadow.py`)

Every `/predict` and `/predict/batch` call:
1. Runs the **Production** model → this is what the caller receives
2. If a Staging model is loaded, runs it silently in parallel
3. Appends both labels to an in-memory log

The MLOps dashboard reads this log via `GET /shadow/log` to populate the A/B comparison tile.

**Current limitation:** the log is in-memory and clears on container restart. The swap point to Postgres is clearly marked in `shadow.py` with a TODO — waiting on Charlie/Ha's `predictions` table schema.

---

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `MLFLOW_TRACKING_URI` | *(unset)* | MLflow server URL. If unset, falls back to pickle |
| `MODEL_NAME` | *(unset)* | Registered model name in MLflow (e.g. `sentiment-baseline`) |
| `MODEL_STAGE` | `Production` | MLflow stage to load (`Production` or `Staging`) |
| `MODEL_PICKLE_PATH` | `models/artifacts/baseline.pkl` | Fallback pickle path |
| `ADMIN_TOKEN` | *(unset)* | Token required by `/reload`. If unset, endpoint is disabled |
| `API_URL` | `http://localhost:8000` | Used by the dashboard to reach this service |

---

## How to run locally

### With Docker Compose (recommended)
```bash
docker compose up -d api
```

### Standalone
```bash
pip install -r api/requirements.txt
uvicorn api.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Quick smoke test
```bash
# Health check
curl http://localhost:8000/health

# Single prediction
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "The coffee was amazing!"}'

# Batch prediction
curl -X POST http://localhost:8000/predict/batch \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Great service", "Waited too long", "It was okay"]}'
```

Interactive docs are available at `http://localhost:8000/docs` once the service is running.

---

## Open TODOs

| File | TODO | Owner | When |
|---|---|---|---|
| `shadow.py` + `main.py` | Swap in-memory log for `INSERT` into `predictions` Postgres table | Charlie/Ha (schema) → Amelia (swap) | Once predictions table is live |
| `schemas.py` | Add `probabilities: dict[str, float]` to `PredictResponse` | Amelia | Once model supports `predict_proba` reliably (LogReg does; LinearSVC needs `CalibratedClassifierCV`) |
| `model_loader.py` | Switch from MLflow stages to aliases (`models:/name@production`) when upgrading to MLflow ≥ 3 | Amelia | If/when MLflow is upgraded |
| `model_loader.py` | Switch `mlflow.sklearn.load_model` to `mlflow.pyfunc.load_model` when DistilBERT is registered | Van (register model) → Amelia (loader update) | Once Van registers DistilBERT in MLflow |

---

## Notes for teammates

- **Van** — once you register DistilBERT in MLflow and promote a model to `Production`, the API will load it automatically on next startup (or immediately via `POST /reload`). Promote a second candidate to `Staging` and the shadow deploy tile in the MLOps dashboard will start populating without any code changes.
- **Charlie/Ha** — the shadow log swap point is in `shadow.py` lines 6–8. Once your `predictions` table schema is finalised, the change is: replace `_log.append(...)` in `shadow.record()` with a `psycopg2` INSERT using the same fields (`text`, `production_label`, `staging_label`, `stage`).