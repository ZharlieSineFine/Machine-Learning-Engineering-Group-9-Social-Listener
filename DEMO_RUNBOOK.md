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
| **Spike** | Latest batch jumps to **~51% negative**; red "⚠ above threshold" alert; timeline spikes today | `demo_spike` → inference → Airflow drift gate **blocks** → retrain DAG **triggered** |

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
   docker compose build postgres minio mlflow airflow-init dashboard
   ```

---

## The demo — two commands

```powershell
# STEP 0 (optional, first run): bring the whole stack up
docker compose up -d

# STEP 1 — normal day
.\scripts\demo_up.ps1
#   -> starts postgres, minio, mlflow, airflow, dashboard
#   -> generates the replay streams + scores a clean 2-week history
#   -> open http://localhost:8501  (Sentiment normal, ~20% negative)   [~30s]

# STEP 2 — inject the spike
.\scripts\demo_spike.ps1
#   -> champion model scores today's negative-review burst  (negative% jumps to ~51%)
#   -> Airflow evaluate_and_monitor detects drift (score 1.0), blocks the gate,
#      flags monitoring_reports.triggered_retrain, and triggers medallion_train_cycle
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
                                 └─ spike only ─► Airflow evaluate_and_monitor
                                                    ├─ Evidently drift gate (blocked)
                                                    ├─ monitoring_reports row (+ MinIO HTML)
                                                    └─ TriggerDagRunOperator ─► medallion_train_cycle (retrain)
```

- **Inference** (`serving/batch_infer.py`): loads the champion pickle, applies the
  tuned negative threshold (0.46), maps classes to `negative/neutral/positive`, and
  writes predictions into the `reviews` table. The spike batch is stamped "today".
- **Dashboard** (`dashboard/app.py`): KPI tiles + the spike alert summarise the
  **latest batch**; the timeline shows the multi-day trend. Alert fires when the
  latest batch is ≥ 25% negative. Auto-refreshes every ~5s during the demo.
- **Monitoring** (`airflow/dags/evaluate_and_monitor.py` + `monitoring/drift_checks.py`):
  with `DRIFT_REPLAY_SCENARIO=spike` the gate compares the spike-day batch vs the
  holdout baseline, detects target drift, and fires the retrain DAG.

---

## Do we need MLflow and FastAPI?

**Not on the critical path.** The dashboard reads predictions straight from Postgres,
and Airflow runs the drift→retrain loop. So the demo needs only **Postgres + batch
inference + Airflow + the dashboard**.

- **MLflow** runs alongside as the model registry / run-history view on the MLOps
  monitor page. Inference loads the champion pickle directly (faster, no dependency).
- **FastAPI** is an optional live-serving lane. It's left out of the demo bring-up
  because it needs a registered/string-label model; the marketing dashboard doesn't
  use it. To enable it later: register the champion in MLflow (or mount the pickle +
  map integer classes to labels), then `docker compose up -d api`.

---

## URLs

| Service | URL | Notes |
|---|---|---|
| Dashboard | http://localhost:8501 | Marketing view + MLOps Monitor page |
| Airflow | http://localhost:8080 | airflow / airflow |
| MLflow | http://localhost:5001 | registry / runs |
| MinIO console | http://localhost:9001 | minioadmin / minioadmin |

---

## Troubleshooting

- **Dashboard didn't update** — it auto-refreshes every ~5s; otherwise refresh the
  browser (the data cache TTL is 5s).
- **Reset to normal** — just re-run `.\scripts\demo_up.ps1`.
- **Airflow step errors** — confirm the scheduler is up:
  `docker exec sentiment-airflow-scheduler airflow version`.
- **Stop everything** — `docker compose down` (data persists in named volumes).
