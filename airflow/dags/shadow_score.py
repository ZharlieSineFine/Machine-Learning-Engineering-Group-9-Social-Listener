"""Airflow DAG — batch inference on recent reviews (6-hour cycle).

Scores reviews ingested in the lookback window with the Production model and
optional Staging shadow candidate, writing rows to ``predictions``.

Owner: Amelia (+ Charlie/Ha for scheduling).
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

from models.batch_score import run_batch_score


def _task_shadow_score(**_context) -> dict:
    lookback = int(os.getenv("SHADOW_SCORE_LOOKBACK_HOURS", "6"))
    summary = run_batch_score(lookback_hours=lookback)
    print(f"[shadow_score] {summary}")
    return summary


with DAG(
    dag_id="shadow_score",
    description="Batch-score recent reviews; log Production + Staging predictions",
    start_date=datetime(2025, 1, 1),
    schedule="0 */6 * * *",
    catchup=False,
    default_args={
        "owner": "serving",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["inference", "shadow", "phase2"],
) as dag:
    PythonOperator(
        task_id="score_recent_reviews",
        python_callable=_task_shadow_score,
    )
