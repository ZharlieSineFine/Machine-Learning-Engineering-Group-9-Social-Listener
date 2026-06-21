"""Airflow DAG — every-6h medallion pipeline (bronze → silver → GE → gold).

Each layer is its own task so retries are granular, per-layer status/duration
shows up in the graph, and the GE gate is a visible node (it goes red when
validation blocks promotion). The task bodies are thin wrappers around the pure
functions in ``data.run_daily`` / ``data.refine`` — the building blocks are
unchanged and still callable via ``python -m data.run_daily`` for CLI and tests.

Affected ``review_date`` keys flow silver → ge_gate → gold → publish over XCom.
The final ``publish`` task mirrors the built partitions to MinIO + upserts
Postgres (``data.publish.publish_run``); it is env-gated and no-ops cleanly when
those services aren't configured. Raw dataset paths come from the container env
(see docker-compose ``YELP_TAR_PATH`` / ``TRIPADVISOR_CSV_PATH``).

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
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator

from data.ingest.ingest_reviews import DEFAULT_SILVER_RECENT_YEARS
from data.refine.build_gold import process_review_dates
from data.refine.build_silver import (
    process_ingestion_to_silver,
    read_silver_partition,
    silver_partition_path,
)
from data.run_daily import _run_bronze, validate_silver_partitions

_SOURCES = ["yelp", "tripadvisor"]
_BRONZE_ROOT = _REPO_ROOT / "data" / "bronze"
_SILVER_ROOT = _REPO_ROOT / "data" / "silver" / "reviews"
_GOLD_ROOT = _REPO_ROOT / "data" / "gold"


def _task_bronze(**context) -> None:
    run_date = context["ds"]  # YYYY-MM-DD from Airflow logical date
    _run_bronze(run_date, _SOURCES, _BRONZE_ROOT)
    print(f"[run_daily.bronze] {run_date}: landed {_SOURCES}")


def _task_silver(**context) -> list[str]:
    run_date = context["ds"]
    affected = process_ingestion_to_silver(
        _BRONZE_ROOT,
        _SILVER_ROOT,
        [run_date],
        _SOURCES,
        recent_years=DEFAULT_SILVER_RECENT_YEARS,
    )
    affected = sorted(affected)
    print(f"[run_daily.silver] {run_date}: {len(affected)} review_date partition(s)")
    return affected  # -> XCom


def _task_ge_gate(**context) -> list[str]:
    affected = context["ti"].xcom_pull(task_ids="silver") or []
    if not affected:
        print("[run_daily.ge_gate] no affected partitions; skipping validation")
        return affected
    validate_silver_partitions(_SILVER_ROOT, affected)  # raises DailyRunError on violation
    print(f"[run_daily.ge_gate] validated {len(affected)} partition(s)")
    return affected


def _task_gold(**context) -> dict:
    affected = context["ti"].xcom_pull(task_ids="ge_gate") or []
    if not affected:
        print("[run_daily.gold] no affected partitions; skipping gold build")
        return {"review_dates": [], "total_silver_rows": 0}

    process_review_dates(_SILVER_ROOT, _GOLD_ROOT, affected)

    counts = {
        key: len(read_silver_partition(silver_partition_path(_SILVER_ROOT, key)))
        for key in affected
    }
    total = sum(counts.values())
    print(
        f"[run_daily.gold] {total} silver rows across "
        f"{len(affected)} review_date partition(s)"
    )
    return {"review_dates": affected, "silver_row_counts": counts, "total_silver_rows": total}


def _task_publish(**context) -> dict:
    """Publish the affected partitions to MinIO + Postgres (task 3).

    Env-gated via ``data.publish.publish_run``: a clean no-op when MinIO/Postgres
    aren't configured, so the DAG stays green in environments without them.
    """
    info = context["ti"].xcom_pull(task_ids="gold") or {}
    affected = info.get("review_dates") or []
    if not affected:
        print("[run_daily.publish] no affected partitions; nothing to publish")
        return {}
    from data.publish import publish_run  # lazy: boto3/psycopg2 only needed here

    run_date = context["ds"]
    summary = publish_run(
        sorted(affected),
        run_date,
        bronze_root=_BRONZE_ROOT,
        silver_root=_SILVER_ROOT,
        gold_root=_GOLD_ROOT,
    )
    print(f"[run_daily.publish] {summary}")
    return summary or {}


with DAG(
    dag_id="run_daily_medallion",
    description="Bronze → Silver → GE gate → Gold for Yelp + TripAdvisor",
    start_date=datetime(2025, 1, 1),
    schedule="0 */6 * * *",  # every 6h, per ARCHITECTURE.md §3 batch cycle
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["data", "medallion"],
) as dag:
    # Course-style stage sentinels (cf. lab_5/dags/dag.py): an explicit source-check
    # entry node and a terminal "completed" node make the medallion stages legible in
    # the graph and give downstream DAGs a single node to depend on. EmptyOperator is
    # the Airflow 2.9 successor to the lab's DummyOperator.
    dep_check_source_data = EmptyOperator(task_id="dep_check_source_data")
    bronze = PythonOperator(task_id="bronze", python_callable=_task_bronze)
    silver = PythonOperator(task_id="silver", python_callable=_task_silver)
    ge_gate = PythonOperator(task_id="ge_gate", python_callable=_task_ge_gate)
    gold = PythonOperator(task_id="gold", python_callable=_task_gold)
    publish = PythonOperator(task_id="publish", python_callable=_task_publish)
    medallion_completed = EmptyOperator(task_id="medallion_completed")

    (
        dep_check_source_data
        >> bronze
        >> silver
        >> ge_gate
        >> gold
        >> publish
        >> medallion_completed
    )
