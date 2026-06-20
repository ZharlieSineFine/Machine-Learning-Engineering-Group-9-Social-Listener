# Airflow — Orchestration

**Owner (DAGs):** Charlie + Ha
**Owner (infra):** Anh

Airflow runs the DAGs that move data through the pipeline. The Airflow container itself is wired up in `docker-compose.yml`.

## DAGs

| File | Schedule | Purpose | Owner |
|---|---|---|---|
| `dags/full_cycle.py` | `@daily` | **Full cycle in one DAG:** ingest → bronze → silver → GE → gold → train → promote → reload API → Evidently monitor | Charlie, Ha, Van |
| `dags/ingest_reviews.py` | `@daily` | Pull product reviews → Postgres `reviews` table, validated by Great Expectations | Charlie, Ha |
| `dags/run_daily.py` | `@daily` | Medallion only: bronze → silver → GE gate → gold | Charlie, Ha |
| `dags/train_model.py` | `@weekly` | Kick off `models/train.py`, log run to MLflow, register best model | Van |
| `dags/evaluate_and_monitor.py` | `@daily` | Run Evidently drift checks; gate promotion to `Production` | Charlie, Ha |

## Full cycle DAG (`medallion_train_cycle`)

`dags/full_cycle.py` chains the whole loop end to end so one trigger produces a deployable model:

```
ingest >> bronze >> silver >> ge_gate >> gold >> train >> promote >> reload_api >> monitor
```

- **train** reads the Gold feature/label stores via `models/gold_loader.py` (joins them on `review_id`, materializes a CSV, then calls the unchanged `models/train.py`). While real Gold data is still being wired up, an empty Gold store falls back to the sample CSV automatically — swap the source in `gold_loader.load_gold_training_frame` when the real DB lands.
- **promote** (`models/promote.py`) transitions the freshly registered MLflow version to the `Production` stage, gated on `f1_macro`. The FastAPI service only serves `Production`, so this is what makes a new model deployable.
- **reload_api** POSTs `/reload` to the API (needs `ADMIN_TOKEN`) so the running service picks up the new model without a restart. Best-effort — a flaky reload won't fail the cycle.
- **monitor** runs the Evidently drift report at the end of the cycle (same logic as `evaluate_and_monitor`): uploads the HTML report to `s3://monitoring/{run_date}/report.html` and records a pointer row in `monitoring_reports` for the dashboard.

**Avoid double-runs:** `medallion_train_cycle` overlaps with the standalone `run_daily_medallion`, `train_model`, and `evaluate_and_monitor` DAGs. Pause those in the Airflow UI so each step only runs once (inside the cycle).

## Local notes

- Airflow webserver: `http://localhost:8080` (default user `airflow` / `airflow`, change in `.env`)
- DAGs auto-load from `airflow/dags/` — no restart needed after edit
- Use the `BashOperator` to call `models/train.py` so training code stays portable
- Heavy lifts (transformer fine-tuning) should be `KubernetesPodOperator` in cloud, but `PythonOperator` is fine locally
