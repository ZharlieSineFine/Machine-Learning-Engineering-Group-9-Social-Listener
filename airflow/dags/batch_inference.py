"""Airflow DAG — scheduled batch inference (every 6h).

The "production heartbeat": every 6 hours it scores the latest window of reviews
with the champion model and writes the predictions into the Postgres ``reviews``
table the dashboard reads. This is what makes the dashboard's "batch inference
every 6 hours" line literally true between manual demo runs.

    latest replay window (text) -> champion model -> labels -> reviews -> dashboard

A guard (``ShortCircuitOperator``) skips the scheduled run when either:

  * ``INFERENCE_PAUSED=1`` is set — a manual kill-switch (e.g. while presenting), or
  * a negative-review spike is already live — the latest batch in ``reviews`` is at
    or above ``INFERENCE_SPIKE_GUARD_PCT`` (default 40%) negative. This stops an
    automated run from overwriting a spike you just injected / an active alert the
    on-call team is looking at. Set the threshold above 100 to disable this branch.

The inference itself reuses ``serving.batch_infer.run`` — the same code path the
demo scripts call. Imports are deferred into the task callables so DAG parsing
stays light and a missing optional dep can't break the whole scheduler.

Owner: Charlie + Ha (Data & Eval).
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
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# Latest-batch negative share (0..100) at or above which a spike is considered
# "live" and the scheduled run is skipped so it can't clobber the alert/demo.
# Set > 100 to disable the auto-skip and rely only on INFERENCE_PAUSED.
SPIKE_GUARD_PCT = float(os.getenv("INFERENCE_SPIKE_GUARD_PCT", "40"))


def _app_dsn() -> str:
    user = os.environ["POSTGRES_USER"]
    pw = os.environ["POSTGRES_PASSWORD"]
    host = os.environ.get("POSTGRES_HOST", "postgres")
    port = os.environ.get("POSTGRES_PORT", "5432")
    db = os.environ["POSTGRES_DB"]
    return f"postgresql://{user}:{pw}@{host}:{port}/{db}"


def _latest_batch_negative_pct() -> tuple[float, int]:
    """(% negative, row count) of the most recent ingested day in ``reviews``.

    Mirrors the dashboard's ``latest_batch`` definition (most recent ingested day)
    so the guard reasons over exactly what the audience sees. Returns (0.0, 0)
    when the table is empty.
    """
    import psycopg2

    conn = psycopg2.connect(_app_dsn())
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT COALESCE(100.0 * COUNT(*) FILTER (WHERE label = 'negative') "
                "       / NULLIF(COUNT(*), 0), 0), "
                "       COUNT(*) "
                "FROM reviews "
                "WHERE ingested_at::date = (SELECT MAX(ingested_at)::date FROM reviews)"
            )
            pct, n = cur.fetchone()
    finally:
        conn.close()
    return float(pct or 0.0), int(n or 0)


def _should_run_inference(**context) -> bool:
    """ShortCircuit guard: run inference unless paused or a spike/alert is live."""
    if os.getenv("INFERENCE_PAUSED") == "1":
        print("[batch_inference.guard] INFERENCE_PAUSED=1 -> skipping scheduled run")
        return False

    neg_pct, n = _latest_batch_negative_pct()
    if n > 0 and neg_pct >= SPIKE_GUARD_PCT:
        print(
            f"[batch_inference.guard] live spike detected "
            f"(latest batch {neg_pct:.1f}% negative >= {SPIKE_GUARD_PCT:.0f}% guard) "
            f"-> skipping so the run doesn't clobber the alert/demo"
        )
        return False

    print(f"[batch_inference.guard] clear to run (latest batch {neg_pct:.1f}% negative, n={n})")
    return True


def _task_score(**context) -> dict:
    """Score the latest replay batch into ``reviews`` (the dashboard heartbeat).

    Replaces only the most recent day (``clear_today``) and stamps the rows "now"
    (``as_now``), so the multi-day history/timeline is preserved while today's
    batch is refreshed with freshly-scored reviews.
    """
    from serving.batch_infer import run

    summary = run(
        scenario="stable",
        n_recent=1,
        as_now=True,
        clear_today=True,
    )
    print(f"[batch_inference] {summary}")
    return summary


with DAG(
    dag_id="batch_inference",
    description="Scheduled champion inference -> reviews (guarded against clobbering a live spike)",
    start_date=datetime(2025, 1, 1),
    schedule="0 */6 * * *",  # every 6h, per ARCHITECTURE.md §3 batch cycle
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["inference", "serving"],
) as dag:
    guard = ShortCircuitOperator(
        task_id="guard_not_paused_or_spiking",
        python_callable=_should_run_inference,
    )
    score = PythonOperator(
        task_id="score_latest_batch",
        python_callable=_task_score,
    )
    inference_completed = EmptyOperator(task_id="inference_completed")

    guard >> score >> inference_completed
