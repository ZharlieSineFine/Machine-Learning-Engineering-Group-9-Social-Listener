"""Airflow DAG — daily medallion pipeline (bronze → silver → GE → gold).

Thin wrapper around ``data.run_daily.run_daily``. Raw dataset paths come from
the container env (see docker-compose ``YELP_TAR_PATH`` / ``TRIPADVISOR_CSV_PATH``).

Owner: Charlie + Ha.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path("/opt/project")
if _REPO_ROOT.exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator

from data.run_daily import run_daily


def _task_run_daily(**context) -> dict:
    run_date = context["ds"]  # YYYY-MM-DD from Airflow logical date
    summary = run_daily(
        run_date,
        ["yelp", "tripadvisor"],
        bronze_root=_REPO_ROOT / "data" / "bronze",
        silver_root=_REPO_ROOT / "data" / "silver" / "reviews",
        gold_root=_REPO_ROOT / "data" / "gold",
    )
    print(
        f"[run_daily] {summary['run_date']}: "
        f"{summary['total_silver_rows']} silver rows, "
        f"{len(summary['review_dates'])} review_date partition(s)"
    )
    return summary


with DAG(
    dag_id="run_daily_medallion",
    description="Bronze → Silver → GE gate → Gold for Yelp + TripAdvisor",
    start_date=datetime(2025, 1, 1),
    schedule="@daily",
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["data", "medallion"],
) as dag:
    PythonOperator(
        task_id="bronze_silver_gold",
        python_callable=_task_run_daily,
    )
