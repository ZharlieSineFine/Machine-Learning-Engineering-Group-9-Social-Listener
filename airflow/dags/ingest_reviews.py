"""Airflow DAG — daily ingestion of the review CSV into Postgres.

Thin wrapper around `data.ingest.ingest_reviews.ingest`. All real logic
lives in the pure module so it stays unit-testable without Airflow.

Owner: Charlie + Ha.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

# /opt/project is the in-container mount of the repo root (see docker-compose).
# Adding it to sys.path lets `from data.ingest...` resolve from the DAG.
_REPO_ROOT = Path("/opt/project")
if _REPO_ROOT.exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator

from data.ingest.ingest_reviews import _default_csv_path, _default_dsn, ingest


def _task_ingest(**_context) -> int:
    n = ingest(_default_csv_path(), _default_dsn(), truncate=True)
    print(f"[ingest_reviews] wrote {n} rows to reviews")
    return n


with DAG(
    dag_id="ingest_reviews",
    description="Load the sample review CSV into Postgres `reviews`",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=5),
    },
    tags=["data", "phase1"],
) as dag:
    PythonOperator(
        task_id="ingest_csv_to_postgres",
        python_callable=_task_ingest,
    )
