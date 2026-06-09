"""Airflow DAG — daily Evidently drift report on the `reviews` table.

Phase 1 stub: split the table 80/20 by ingestion time into reference + current,
run Evidently, drop the report in MinIO, pointer row in Postgres. No blocking
behavior yet — that lands in Step 10 (Phase 2).

Owner: Charlie + Ha.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path("/opt/project")
if _REPO_ROOT.exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator


def _split_reviews(dsn: str):
    """Pull all rows from `reviews`, split 80/20 by ingested_at."""
    import pandas as pd
    import psycopg2

    conn = psycopg2.connect(dsn)
    try:
        df = pd.read_sql(
            "SELECT text, label, rating, source FROM reviews ORDER BY ingested_at",
            conn,
        )
    finally:
        conn.close()
    if len(df) < 10:
        raise ValueError(f"Not enough rows for drift check (have {len(df)}, need >= 10)")
    cut = int(len(df) * 0.8)
    return df.iloc[:cut].reset_index(drop=True), df.iloc[cut:].reset_index(drop=True)


def _load_production_model():
    """Load the current Production model from MLflow. Returns None on miss."""
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")
    name = os.environ.get("MODEL_NAME", "sentiment-baseline")
    stage = os.environ.get("MODEL_STAGE", "Production")
    if not tracking_uri:
        return None
    try:
        import mlflow
        import mlflow.sklearn
        mlflow.set_tracking_uri(tracking_uri)
        return mlflow.sklearn.load_model(f"models:/{name}/{stage}")
    except Exception as exc:
        print(f"[evaluate_and_monitor] could not load model: {exc}; running drift only")
        return None


def _task_drift(**_context) -> dict:
    import boto3
    import psycopg2
    from botocore.client import Config

    from monitoring.drift_checks import _summary_for_log, evaluate

    user = os.environ["POSTGRES_USER"]
    pw = os.environ["POSTGRES_PASSWORD"]
    host = os.environ["POSTGRES_HOST"]
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ["POSTGRES_DB"]
    dsn = f"postgresql://{user}:{pw}@{host}:{port}/{db}"

    reference_df, current_df = _split_reviews(dsn)

    minio = boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )

    model = _load_production_model()

    conn = psycopg2.connect(dsn)
    conn.autocommit = False
    try:
        result = evaluate(
            reference_df, current_df, conn, minio,
            model=model,
            raise_on_block=True,  # task fails red so Airflow blocks the promotion DAG
        )
        conn.commit()
    finally:
        conn.close()

    print(f"[evaluate_and_monitor] {_summary_for_log(result)}")
    return result


with DAG(
    dag_id="evaluate_and_monitor",
    description="Daily Evidently drift report for the `reviews` table",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["monitoring", "phase1"],
) as dag:
    PythonOperator(
        task_id="compute_and_log_drift",
        python_callable=_task_drift,
    )
