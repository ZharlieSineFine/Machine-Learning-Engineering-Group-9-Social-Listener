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

MONITORING_BUCKET = "monitoring"


def _minio_client():
    """boto3 S3 client pointed at MinIO, from the same env the stack already sets."""
    import boto3
    from botocore.client import Config

    return boto3.client(
        "s3",
        endpoint_url=os.environ["MLFLOW_S3_ENDPOINT_URL"],
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        config=Config(signature_version="s3v4"),
        region_name="us-east-1",
    )


def _app_dsn() -> str:
    user = os.environ["POSTGRES_USER"]
    pw = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ["POSTGRES_DB"]
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def _upload_report(client, local_path: Path, run_date: str) -> str:
    """Upload the HTML report to s3://monitoring/{run_date}/report.html."""
    key = f"{run_date}/report.html"
    client.upload_file(
        Filename=str(local_path),
        Bucket=MONITORING_BUCKET,
        Key=key,
        ExtraArgs={"ContentType": "text/html"},
    )
    return f"s3://{MONITORING_BUCKET}/{key}"


def _record_report(dsn: str, run_date: str, report_url: str, result) -> None:
    """Insert a pointer row into `monitoring_reports` for the dashboard to read."""
    import psycopg2

    conn = psycopg2.connect(dsn)
    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO monitoring_reports "
                    "(run_date, report_type, report_url, drift_score, blocked_promotion) "
                    "VALUES (%s, %s, %s, %s, %s)",
                    (
                        run_date,
                        "data_drift",
                        report_url,
                        float(result.drift_score),
                        not result.passed_gate,
                    ),
                )
    finally:
        conn.close()


def _task_drift(**context) -> dict:
    from dataclasses import asdict

    from monitoring.drift_checks import run_drift_check

    run_date = context["ds"]  # YYYY-MM-DD logical date

    result = run_drift_check()

    report_url = _upload_report(_minio_client(), Path(result.report_path), run_date)
    _record_report(_app_dsn(), run_date, report_url, result)

    print(
        f"[evaluate_and_monitor] {run_date}: drift_score={result.drift_score:.3f} "
        f"passed_gate={result.passed_gate} evidently_ran={result.evidently_ran} "
        f"-> {report_url}"
    )
    return {**asdict(result), "report_url": report_url, "run_date": run_date}


with DAG(
    dag_id="evaluate_and_monitor",
    description="Every-6h Evidently drift report → MinIO + monitoring_reports",
    start_date=datetime(2025, 1, 1),
    schedule="0 */6 * * *",  # every 6h, per ARCHITECTURE.md §3 batch cycle
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