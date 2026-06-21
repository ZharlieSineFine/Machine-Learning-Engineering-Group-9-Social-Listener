"""Airflow DAG — pure-observation drift monitor (every 6h).

A read-only **observer** that sits next to ``medallion_pipeline``: it compares the
recent review window against the reference, writes an Evidently report + a
``monitoring_reports`` pointer row (which the dashboard reads), and — when drift
blocks the gate — fires ``send_alert``. That's it. There is **no**
``TriggerDagRunOperator`` and no side effect on the data or the model: drift
alerts a human, who decides whether to kick an off-cycle retrain
(``medallion_pipeline`` with ``FORCE_TRAIN=1``).

Kept separate from the build pipeline on purpose:
  * **Failure isolation** — an Evidently hiccup here goes red as a *monitoring*
    failure, not as a "data pipeline failed" page.
  * **Pure observation** — read-only, its own 6h cadence, no dependency on the
    producing DAG having just run.

Default source is the observational silver window (``run_drift_check``). Set
``DRIFT_REPLAY_SCENARIO=spike`` (with optional ``DRIFT_REPLAY_ASOF`` /
``DRIFT_REPLAY_N_RECENT``) to drive the gate from the replay simulator — the demo
path where the negative-review spike trips the gate.

Owner: Charlie + Ha (Monitoring).
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# /opt/project is the in-container mount of the repo root (see docker-compose).
_REPO_ROOT = Path("/opt/project")
if _REPO_ROOT.exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator, ShortCircuitOperator

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
    """Run the Evidently drift check, persist the report + a monitoring_reports row.

    Default source is the observational silver window. Set ``DRIFT_REPLAY_SCENARIO``
    (stable|spike) to drive the gate from the replay simulator instead — the demo
    path where the spike trips the gate. Optional ``DRIFT_REPLAY_ASOF`` /
    ``DRIFT_REPLAY_N_RECENT`` select the current window.
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
    return {
        "drift_score": drift_score,
        "blocked": blocked,
        "evidently_ran": evidently_ran,
        "n_reference": n_ref,
        "n_current": n_cur,
        "report_url": report_url,
        "run_date": run_date,
        "source": source,
    }


def _should_alert(**context) -> bool:
    """ShortCircuit: only fire the alert when drift actually blocked the gate."""
    info = context["ti"].xcom_pull(task_ids="drift_check") or {}
    blocked = bool(info.get("blocked"))
    print(
        f"[evaluate_and_monitor] should_alert={blocked} "
        f"(drift_score={info.get('drift_score')})"
    )
    return blocked


def _task_send_alert(**context) -> None:
    """Surface the drift alert.

    The ``monitoring_reports`` row written by ``drift_check`` is the durable signal
    the dashboard renders; this task makes the alert loud in the Airflow logs so a
    human notices and can decide whether to trigger an off-cycle retrain
    (``medallion_pipeline`` with ``FORCE_TRAIN=1``). No automatic retrain fires.
    """
    info = context["ti"].xcom_pull(task_ids="drift_check") or {}
    print(
        "[evaluate_and_monitor.ALERT] *** DATA DRIFT DETECTED *** "
        f"run_date={info.get('run_date')} drift_score={info.get('drift_score')} "
        f"report={info.get('report_url')} — review on the dashboard; retrain "
        "off-cycle with FORCE_TRAIN=1 on medallion_pipeline if warranted."
    )


with DAG(
    dag_id="evaluate_and_monitor",
    description="Pure-observation Evidently drift monitor → monitoring_reports + alert (no retrain)",
    start_date=datetime(2025, 1, 1),
    schedule="0 */6 * * *",  # every 6h, per ARCHITECTURE.md §3 batch cycle
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["monitoring", "observation"],
) as dag:
    drift_check = PythonOperator(
        task_id="drift_check",
        python_callable=_task_drift,
    )
    should_alert = ShortCircuitOperator(
        task_id="should_alert",
        python_callable=_should_alert,
    )
    send_alert = PythonOperator(
        task_id="send_alert",
        python_callable=_task_send_alert,
    )

    drift_check >> should_alert >> send_alert
