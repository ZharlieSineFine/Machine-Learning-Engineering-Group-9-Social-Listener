"""Airflow DAG — production batch inference (runs when fresh data lands).

The read-side serving step: each time the medallion publishes a fresh data build,
score the latest silver window with the champion model and write the predictions
into the Postgres ``reviews`` table the dashboard reads.

    medallion publish ──(Dataset)──▶ score latest silver ─▶ champion ─▶ reviews ─▶ dashboard

Scheduling is **data-aware**: the DAG is triggered by the ``reviews_gold`` Dataset
that ``medallion_pipeline``'s ``publish`` task updates — not a cron. This keeps a
*real* data dependency (inference scores what was just built) while staying a
**separate** DAG with its own retries/failure domain, so a data-build failure can't
take serving down. No DAG imperatively triggers another — the coupling is the
declarative Dataset.

``INFERENCE_PAUSED=1`` is a manual kill-switch to pause serving (e.g. during an
incident) without unpausing the DAG.

The inference itself reuses ``serving.batch_infer.run_on_silver`` — the same pure
code path the CLI/tests use. Imports are deferred into the task callables so DAG
parsing stays light and a missing optional dep can't break the whole scheduler.

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
from airflow.datasets import Dataset
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator

# Declarative data dependency: medallion_pipeline.publish updates this Dataset; this
# DAG is scheduled on it. Datasets are keyed by URI, so this string MUST match the
# outlet declared in airflow/dags/medallion_pipeline.py.
REVIEWS_GOLD_DATASET = Dataset("postgres://app/reviews_gold")


def _should_run_inference(**context) -> bool:
    """ShortCircuit kill-switch: run unless serving is manually paused."""
    if os.getenv("INFERENCE_PAUSED") == "1":
        print("[batch_inference.guard] INFERENCE_PAUSED=1 -> skipping run")
        return False
    print("[batch_inference.guard] clear to run")
    return True


def _task_score(**context) -> dict:
    """Score the latest silver window into ``reviews`` (production read-side).

    Replaces only the most recent day (``clear_today``) and stamps the rows "now"
    (``as_now``), so the multi-day history/timeline is preserved while today's batch
    is refreshed with freshly-scored reviews.
    """
    from serving.batch_infer import run_on_silver

    summary = run_on_silver(as_now=True, clear_today=True)
    print(f"[batch_inference] {summary}")
    return summary


with DAG(
    dag_id="batch_inference",
    description="Champion inference on the freshly-built silver window -> reviews (Dataset-triggered by medallion publish)",
    start_date=datetime(2025, 1, 1),
    schedule=[REVIEWS_GOLD_DATASET],  # data-aware: runs when the medallion publishes
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["inference", "serving"],
) as dag:
    guard = ShortCircuitOperator(
        task_id="guard_not_paused",
        python_callable=_should_run_inference,
    )
    score = PythonOperator(
        task_id="score_latest_batch",
        python_callable=_task_score,
    )
    inference_completed = EmptyOperator(task_id="inference_completed")

    guard >> score >> inference_completed
