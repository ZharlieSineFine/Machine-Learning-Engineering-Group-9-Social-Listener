from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path("/opt/project")
if _REPO_ROOT.exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator

MONITORING_BUCKET = "monitoring"
RETRAIN_DAG_ID = "medallion_train_cycle"


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


def _load_production_model():
    """Load the current Production model for prediction-distribution drift.

    Mirrors ``api/app/model_loader.py``: MLflow registry first
    (``models:/<MODEL_NAME>/<MODEL_STAGE>``), then the local pickle fallback.
    Returns ``None`` if neither is available — prediction drift then degrades
    gracefully (data + target drift still run).
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("MODEL_NAME")
    if tracking_uri and model_name:
        try:
            import mlflow
            import mlflow.sklearn

            mlflow.set_tracking_uri(tracking_uri)
            stage = os.getenv("MODEL_STAGE", "Production")
            uri = f"models:/{model_name}/{stage}"
            print(f"[evaluate_and_monitor] loading model from MLflow: {uri}")
            return mlflow.sklearn.load_model(uri)
        except Exception as exc:
            print(f"[evaluate_and_monitor] MLflow model load failed ({exc}); trying pickle")

    pickle_path = Path(
        os.getenv(
            "MODEL_PICKLE_PATH",
            "/opt/project/models/artifacts/baseline.pkl",
        )
    )
    if pickle_path.exists():
        import pickle

        print(f"[evaluate_and_monitor] loading model from pickle: {pickle_path}")
        with open(pickle_path, "rb") as fh:
            return pickle.load(fh)

    print("[evaluate_and_monitor] no model available; prediction drift will be skipped")
    return None


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
    """Insert a pointer row into `monitoring_reports` for the dashboard to read.

    Same row shape as before — the combined Evidently HTML carries the
    prediction-drift + PSI detail; ``drift_score`` holds the data-drift score.
    """
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
                        float(result.data_drift_score),
                        bool(result.blocked),
                    ),
                )
    finally:
        conn.close()


def _task_drift(**context) -> dict:
    from monitoring.drift_checks import run_monitor_drift

    run_date = context["ds"]  # YYYY-MM-DD logical date
    dsn = _app_dsn()

    # Score the Production model so prediction-distribution drift is computed;
    # prefer logged predictions for the current window, falling back to silver.
    result = run_monitor_drift(model=_load_production_model(), dsn=dsn)

    report_url = _upload_report(_minio_client(), Path(result.report_path), run_date)
    _record_report(dsn, run_date, report_url, result)

    print(
        f"[evaluate_and_monitor] {run_date}: data_drift={result.data_drift_score:.3f} "
        f"target_drift={result.target_drift_score} pred_drift={result.prediction_drift_score} "
        f"psi_by_column={result.psi_by_column} blocked={result.blocked} "
        f"used_model={result.used_model} evidently_ran={result.evidently_ran} -> {report_url}"
    )
    # Return an XCom-light, JSON-serializable summary (no html bytes).
    return {
        "drift_score": result.data_drift_score,
        "target_drift_score": result.target_drift_score,
        "prediction_drift_score": result.prediction_drift_score,
        "blocked": result.blocked,
        "evidently_ran": result.evidently_ran,
        "n_reference": result.n_reference,
        "n_current": result.n_current,
        "report_url": report_url,
        "run_date": run_date,
    }


def _should_retrain(**context) -> bool:
    """ShortCircuit: only let the retrain trigger run when drift blocked the gate."""
    info = context["ti"].xcom_pull(task_ids="compute_and_log_drift") or {}
    blocked = bool(info.get("blocked"))
    print(
        f"[evaluate_and_monitor] should_retrain={blocked} "
        f"(drift_score={info.get('drift_score')})"
    )
    return blocked


with DAG(
    dag_id="evaluate_and_monitor",
    description="Every-6h Evidently data+target+prediction drift (PSI) → MinIO + monitoring_reports",
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
    compute_and_log_drift = PythonOperator(
        task_id="compute_and_log_drift",
        python_callable=_task_drift,
    )

    # Close the loop: when drift blocks the gate, kick off a full retrain cycle.
    should_retrain = ShortCircuitOperator(
        task_id="should_retrain",
        python_callable=_should_retrain,
    )

    trigger_retrain = TriggerDagRunOperator(
        task_id="trigger_retrain",
        trigger_dag_id=RETRAIN_DAG_ID,
        reset_dag_run=True,        # avoid duplicate run_id clashes on re-fire
        wait_for_completion=False,
    )

    compute_and_log_drift >> should_retrain >> trigger_retrain