# Airflow — Orchestration

**Owner (DAGs):** Charlie + Ha
**Owner (infra):** Anh

Airflow runs four DAGs that move data through the pipeline. The Airflow container itself is wired up in `docker-compose.yml`. The data build runs on a **6-hour batch cycle**; inference is **data-triggered** off the build (Airflow Dataset), so it scores what was just built while staying a separate, isolated DAG. The model is **not** retrained on a schedule — retraining is human-triggered (`FORCE_TRAIN=1`).

## DAGs

| File | Schedule | Purpose | Owner |
|---|---|---|---|
| `dags/medallion_pipeline.py` | `0 */6 * * *` (6h) | Build the medallion (bronze→silver→GE→gold→publish) every 6h. `publish` updates the `reviews_gold` **Dataset** (triggers inference). The `train→gate→promote→reload` branch is short-circuited unless `FORCE_TRAIN=1` (on-demand retrain). | Charlie, Ha, Van |
| `dags/batch_inference.py` | Dataset `reviews_gold` | When the medallion publishes, score the latest silver window with the champion model → write predictions to the Postgres `reviews` table the dashboard reads. `INFERENCE_PAUSED=1` pauses serving. | Charlie, Ha |
| `dags/evaluate_and_monitor.py` | `0 */6 * * *` (6h) | Read-only Evidently drift monitor → `monitoring_reports` + alert. **No auto-retrain** — a human decides whether to run `FORCE_TRAIN=1`. | Charlie, Ha |
| `dags/shadow_deploy_distilbert.py` | manual / triggered | Fine-tune the DistilBERT challenger → register to MLflow `Staging` (never Production). Human-triggered, same as the baseline retrain. | Van, Charlie, Ha |

## Local notes

- Airflow webserver: `http://localhost:8080` (default user `airflow` / `airflow`, change in `.env`)
- DAGs auto-load from `airflow/dags/` — no restart needed after edit
- Use the `BashOperator` to call `models/train.py` so training code stays portable
- Heavy lifts (transformer fine-tuning) should be `KubernetesPodOperator` in cloud, but `PythonOperator` is fine locally

## Phase 1 stub
Create minimal DAGs that just print + write a dummy row so the wiring is testable before any real ingestion code lands.
