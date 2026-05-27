# Tests — Cross-cutting & Integration

**Owner:** Amelia (with each folder owner contributing tests for their code)

## Layout

- **Unit tests live next to the code they test** — `models/test_train.py`, `api/app/test_main.py`, etc.
- **This `tests/` folder is for integration only** — tests that need multiple services running.

## Running

```bash
# Unit only (fast, no docker)
pytest -m "not integration"

# Full suite (boots docker compose)
pytest
```

## What goes here

| File | Purpose |
|---|---|
| `test_e2e_smoke.py` | Stack up → ingest → train → predict → assert response |
| `test_promotion_gate.py` | Poison a batch; assert the eval DAG blocks promotion |
| `test_dashboard_loads.py` | Hit `:8501`, assert HTTP 200 and key elements render |

Mark every test here with `@pytest.mark.integration` so the fast CI lane skips them.
