"""Airflow DAG — every-6h Evidently drift monitor + closed-loop retrain trigger.

Default source is the observational silver window (``run_drift_check``). Set
``DRIFT_REPLAY_SCENARIO=spike`` (with optional ``DRIFT_REPLAY_ASOF`` /
``DRIFT_REPLAY_N_RECENT``) to drive the gate from the replay simulator's output —
the demo path where the negative-review spike trips the gate and fires the
``medallion_train_cycle`` retrain via ``TriggerDagRunOperator``.

Owner: Charlie + Ha (Monitoring).
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


def _record_report(
    dsn: str, run_date: str, report_url: str, drift_score: float, blocked: bool
) -> None:
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
                    (run_date, "data_drift", report_url, float(drift_score), bool(blocked)),
                )
    finally:
        conn.close()


def _task_drift(**context) -> dict:
    """Run the drift check, persist the report, return a JSON-light summary.

    Default source is the observational silver window. Set
    ``DRIFT_REPLAY_SCENARIO`` (stable|spike) to drive the gate from the replay
    simulator instead — the demo path where the spike trips the gate. Optional
    ``DRIFT_REPLAY_ASOF`` / ``DRIFT_REPLAY_N_RECENT`` select the current window.
    """
    run_date = context["ds"]  # YYYY-MM-DD logical date

    scenario = os.getenv("DRIFT_REPLAY_SCENARIO")
    if scenario:
        from monitoring.drift_checks import run_replay_monitor

        n_recent = os.getenv("DRIFT_REPLAY_N_RECENT")
        mon = run_replay_monitor(
            scenario,
            asof=os.getenv("DRIFT_REPLAY_ASOF") or None,
            n_recent=int(n_recent) if n_recent else None,
        )
        drift_score, blocked = mon.data_drift_score, mon.blocked
        report_path, evidently_ran = mon.report_path, mon.evidently_ran
        n_ref, n_cur = mon.n_reference, mon.n_current
        source = f"replay:{scenario}"
    else:
        from monitoring.drift_checks import run_drift_check

        result = run_drift_check()
        drift_score, blocked = result.drift_score, (not result.passed_gate)
        report_path, evidently_ran = result.report_path, result.evidently_ran
        n_ref, n_cur = result.n_reference, result.n_current
        source = "silver"

    report_url = _upload_report(_minio_client(), Path(report_path), run_date)
    _record_report(_app_dsn(), run_date, report_url, drift_score, blocked)

    print(
        f"[evaluate_and_monitor] {run_date} ({source}): drift_score={drift_score:.3f} "
        f"blocked={blocked} evidently_ran={evidently_ran} -> {report_url}"
    )
    # Return an XCom-light, JSON-serializable summary (no html bytes).
    return {
        "drift_score": drift_score,
        "passed_gate": not blocked,
        "blocked": blocked,
        "evidently_ran": evidently_ran,
        "n_reference": n_ref,
        "n_current": n_cur,
        "report_url": report_url,
        "run_date": run_date,
        "source": source,
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


def _task_mark_retrained(**context) -> None:
    """Flag the just-written monitoring_reports row as having triggered a retrain.

    Reuses monitoring.retrain_trigger.mark_triggered_retrain (the same helper as the
    standalone CLI path), which self-heals the schema (ADD COLUMN IF NOT EXISTS) for
    DBs whose volume predates the ``triggered_retrain`` column in init.sql.
    """
    import psycopg2

    from monitoring.retrain_trigger import mark_triggered_retrain

    conn = psycopg2.connect(_app_dsn())
    try:
        ok = mark_triggered_retrain(conn)  # marks the most recent monitoring_reports row
        print(f"[evaluate_and_monitor] marked triggered_retrain={ok}")
    finally:
        conn.close()


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

    # Record that this drift run fired a retrain (task 2's monitoring_reports flag).
    mark_retrained = PythonOperator(
        task_id="mark_retrained",
        python_callable=_task_mark_retrained,
    )

    compute_and_log_drift >> should_retrain >> trigger_retrain >> mark_retrained