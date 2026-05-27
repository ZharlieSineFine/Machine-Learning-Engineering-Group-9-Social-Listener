# Tests — Smoke, Unit & Integration

**Owner:** Amelia (with each folder owner contributing tests for their code)

## Three lanes

| Lane | Where | How to run | Gate |
|---|---|---|---|
| **Smoke** | `tests/test_smoke.py` runs in the dedicated `smoke` compose service | `docker compose run --rm smoke` *or* `pytest tests/test_smoke.py` from a venv | Every PR (fast, no other services) |
| **Unit** | Live next to code — `models/test_train.py`, `api/app/test_main.py`, etc. | `pytest -m "not integration"` | Every PR |
| **Integration** | This `tests/` folder, marked `@pytest.mark.integration` | `pytest -m integration` (boots compose) | PR to `main` |

## Phase 1 smoke test (live, in `test_smoke.py`)

Walks the thin-slice end-to-end **without any other services** (no Postgres, MinIO, MLflow, Airflow). Runs in < 10s on a laptop.

1. `test_sample_csv_contract` — `data/sample/reviews_sample.csv` exists, has > 100 rows, has columns `{text, label, rating, source, restaurant, location}`, and labels are within `LABELS`.
2. `test_baseline_trains_and_predicts` — `models.baseline_sklearn.train(df)` fits and produces an F1 in [0, 1].
3. `test_train_run_writes_artifact` — `models.train.run(data_path, out_path)` writes a pickle and returns `f1_macro` (with `mlflow_run_id=None` in the offline path).
4. `test_health` / `test_predict_positive_and_negative` / `test_predict_rejects_empty` — FastAPI loads the pickle (`MODEL_PICKLE_PATH` env var), `/health` reports `model_source='pickle'`, `/predict` returns a label in `LABELS`, and empty text is rejected with 422.

`conftest.py` makes the project root + `api/` importable so `from models.* import ...` and `from app.* import ...` both resolve.

Every folder owner should add at least one smoke assertion for their layer as it lands.

## Integration tests (Phase 2)

| File | Purpose |
|---|---|
| `test_e2e_smoke.py` | Stack up → ingest → refine → build → train → predict → assert |
| `test_promotion_gate.py` | Poison a batch via replay simulator; assert eval DAG blocks promotion |
| `test_shadow_lane.py` | Stage a candidate model; assert both Production and Staging rows appear in `predictions` |
| `test_dashboard_loads.py` | Hit `:8501`, assert HTTP 200 and key elements render |

Mark every integration test with `@pytest.mark.integration` so CI's fast lane skips them.

## Requirements

- Canonical smoke deps: `infra/docker/smoke/requirements.txt` (used by the `smoke` compose service)
- Local-venv convenience: `tests/requirements.txt` (kept loosely aligned; infra/docker/smoke wins on drift)
