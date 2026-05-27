# Airflow — Orchestration

**Owner (DAGs):** Charlie + Ha
**Owner (infra):** Anh

Airflow runs three core DAGs that move data through the pipeline. The Airflow container itself is wired up in `docker-compose.yml`.

## DAGs

| File | Schedule | Purpose | Owner |
|---|---|---|---|
| `dags/ingest_reviews.py` | `@daily` | Pull product reviews → Postgres `reviews` table, validated by Great Expectations | Charlie, Ha |
| `dags/train_model.py` | `@weekly` | Kick off `models/train.py`, log run to MLflow, register best model | Van |
| `dags/evaluate_and_monitor.py` | `@daily` | Run Evidently drift checks; gate promotion to `Production` | Charlie, Ha |

## Local notes

- Airflow webserver: `http://localhost:8080` (default user `airflow` / `airflow`, change in `.env`)
- DAGs auto-load from `airflow/dags/` — no restart needed after edit
- Use the `BashOperator` to call `models/train.py` so training code stays portable
- Heavy lifts (transformer fine-tuning) should be `KubernetesPodOperator` in cloud, but `PythonOperator` is fine locally

## Phase 1 stub
Create minimal DAGs that just print + write a dummy row so the wiring is testable before any real ingestion code lands.
