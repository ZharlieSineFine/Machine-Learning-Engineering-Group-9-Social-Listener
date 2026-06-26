from __future__ import annotations

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

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

# Downstream link: this DAG emits this Dataset once fresh predictions land, which
# data-triggers evaluate_and_monitor — so drift is checked per batch, right after
# inference. Keyed by URI, so this string MUST match the schedule in
# airflow/dags/evaluate_and_monitor.py.
REVIEWS_PREDICTIONS_DATASET = Dataset("postgres://app/reviews")


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
    description="Champion inference on the freshly-built silver window -> reviews (Dataset-triggered by medallion publish; emits reviews Dataset -> evaluate_and_monitor)",
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
    # outlet: emit the predictions Dataset on success -> data-triggers evaluate_and_monitor.
    inference_completed = EmptyOperator(
        task_id="inference_completed",
        outlets=[REVIEWS_PREDICTIONS_DATASET],
    )

    guard >> score >> inference_completed
