# Final Demo Runbook — BrewLeaf Social Listener

A two-command live demo: a **normal day** on the dashboard, then a **sudden
negative-review spike** that the MLOps backend detects, alerts on, and responds to
with a retrain — all in under a minute.

Set up two windows side by side:
- **Browser** → the Streamlit dashboard (`http://localhost:8501`)
- **Terminal** (Cursor) → where you run the two commands and narrate the backend

---

## What the audience sees

| | Dashboard (browser) | Terminal / backend |
|---|---|---|
| **Normal day** | Sentiment timeline flat at ~18–20% negative; green "Sentiment normal" banner | `demo_up` scores a clean 2-week history with the champion model |
| **Spike** | Latest batch jumps to **~51% negative**; red "⚠ above threshold" alert; timeline spikes today | `demo_spike` → inference → Airflow drift gate **blocks** → **alert recorded** (no auto-retrain) |

---

## Prerequisites (once)

1. **Docker Desktop** running.
2. **Champion model** present at `models/artifacts/baseline.pkl` (the TF-IDF+LogReg
   champion, `champion_baseline_v3.pkl`; ~6 MB, distributed offline — see
   `models/champion_manifest.txt`).
3. **Python venv** at `.venv` (the host runs inference): `scikit-learn`, `pandas`,
   `pyarrow`, `psycopg2`, `sqlalchemy`.
4. Build the images once (a few minutes):
   ```powershell
   docker compose build
   ```

---

## The demo — two commands

```powershell
# STEP 0 (first run only): build the images (a few minutes)
docker compose build

# STEP 1 — normal day
.\scripts\demo_up.ps1
#   -> starts postgres, minio, mlflow, airflow, dashboard, API
#   -> registers the champion in MLflow (Production + Staging), scores a clean
#      2-week history, seeds the shadow-deploy panel
#   -> open http://localhost:8501  (Sentiment normal, ~20% negative)   [~40s]

# STEP 2 — inject the spike
.\scripts\demo_spike.ps1
#   -> champion model scores today's negative-review burst  (negative% jumps to ~51%)
#   -> Airflow evaluate_and_monitor detects drift (score 1.0), blocks the gate, and
#      records the monitoring_reports alert row — NO auto-retrain (human-in-the-loop)
#   -> refresh / watch the dashboard: red alert + today's spike on the timeline  [~15s]
```

(Bash mirrors: `./scripts/demo_up.sh`, `./scripts/demo_spike.sh`.)

Re-run anytime: `demo_up` truncates and re-seeds, so the demo is idempotent.

---

## What happens under the hood

```
replay simulator ──► champion model (batch_infer) ──► reviews table ──► dashboard
   (demo_jun2026                 │                        (Postgres)        (Streamlit)
    stable / spike)              │
                                 └─ spike only ─► Airflow evaluate_and_monitor (read-only observer)
                                                    ├─ Evidently drift gate (blocked)
                                                    ├─ monitoring_reports row (+ MinIO HTML) ─► dashboard alert
                                                    └─ send_alert — human-in-the-loop, NO auto-retrain
                                                       (a human runs medallion_pipeline FORCE_TRAIN=1 if warranted)
```

- **Inference** (`serving/batch_infer.py`): loads the champion pickle, applies the
  tuned negative threshold (0.46), maps classes to `negative/neutral/positive`, and
  writes predictions into the `reviews` table. The spike batch is stamped "today".
- **Dashboard** (`dashboard/app.py`): KPI tiles + the spike alert summarise the
  **latest batch**; the timeline shows the multi-day trend. Alert fires when the
  latest batch is ≥ 25% negative. Auto-refreshes every ~5s during the demo.
- **Monitoring** (`airflow/dags/evaluate_and_monitor.py` + `monitoring/drift_checks.py`):
  with `DRIFT_REPLAY_SCENARIO=spike` the gate compares the spike-day batch vs the
  holdout baseline, detects the drift, blocks the gate, and records the alert row —
  no auto-retrain (a human decides whether to retrain off-cycle with `FORCE_TRAIN=1`).

---

## MLflow + FastAPI (wired in)

The spike→alert path doesn't *depend* on them (the dashboard reads predictions from
Postgres; Airflow runs the drift→alert loop), so they stay off the critical path —
but `demo_up` now brings them up for the full MLOps story:

- **MLflow** is the model registry. `demo_up` registers the champion (idempotently,
  `scripts/register_champion.py`) as `sentiment-baseline` **v1 → Production**
  (neg threshold 0.46) and **v2 → Staging** (0.40, the shadow challenger). Browse them
  at http://localhost:5001. Batch inference still loads the local pickle directly
  (faster); the API loads from the registry.
- **FastAPI** is the live serving lane. `/health` reports `model_source: mlflow`,
  `/predict` returns real string labels, and every call is scored by **both** Production
  and Staging and logged to `/shadow/log` — which feeds the dashboard's **MLOps Monitor
  → Shadow deploy** panel. `demo_up` seeds that panel with a sample batch
  (`scripts/seed_shadow.py`).

---

## URLs

| Service | URL | Notes |
|---|---|---|
| Dashboard | http://localhost:8501 | Marketing view + MLOps Monitor page |
| API (FastAPI) | http://localhost:8000/docs | /health, /predict, /predict/batch, /shadow/log |
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5001 | registry — sentiment-baseline: Production + Staging |
| MinIO console | http://localhost:9001 | minioadmin / minioadmin |

---

## Troubleshooting

- **Dashboard didn't update** — it auto-refreshes every ~5s; otherwise refresh the
  browser (the data cache TTL is 5s).
- **Reset to normal** — just re-run `.\scripts\demo_up.ps1`.
- **Airflow step errors** — confirm the scheduler is up:
  `docker exec sentiment-airflow-scheduler airflow version`.
- **Stop everything** — `docker compose down` (data persists in named volumes).
