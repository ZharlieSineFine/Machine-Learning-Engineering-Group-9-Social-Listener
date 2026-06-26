from __future__ import annotations

import os
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path("/opt/project")
if _REPO_ROOT.exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airflow import DAG
from airflow.datasets import Dataset
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule

REVIEWS_GOLD_DATASET = Dataset("postgres://app/reviews_gold")

from data.ingest.ingest_reviews import DEFAULT_SILVER_RECENT_YEARS
from data.refine.build_gold import process_review_dates
from data.refine.build_silver import (
    process_ingestion_to_silver,
    read_silver_partition,
    silver_partition_path,
)
from data.run_daily import _run_bronze, validate_silver_partitions
from models.gold_loader import materialize_training_csv
from models.promote import promote_to_production
from models.train import run as train_run

_SOURCES = ["yelp", "tripadvisor"]
_BRONZE_ROOT = _REPO_ROOT / "data" / "bronze"
_SILVER_ROOT = _REPO_ROOT / "data" / "silver" / "reviews"
_GOLD_ROOT = _REPO_ROOT / "data" / "gold"
_TRAINING_CSV = _GOLD_ROOT / "_training_frame.csv"


def _minio_client():
    # boto3 S3 client pointed at MinIO.
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


def _task_bronze(**context) -> None:
    run_date = context["ds"]
    _run_bronze(run_date, _SOURCES, _BRONZE_ROOT)
    print(f"[pipeline.bronze] {run_date}: landed {_SOURCES}")


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
    print(f"[pipeline.silver] {run_date}: {len(affected)} review_date partition(s)")
    return affected


def _task_ge_gate(**context) -> list[str]:
    affected = context["ti"].xcom_pull(task_ids="silver") or []
    if not affected:
        print("[pipeline.ge_gate] no affected partitions; skipping validation")
        return affected
    validate_silver_partitions(_SILVER_ROOT, affected) 
    print(f"[pipeline.ge_gate] validated {len(affected)} partition(s)")
    return affected


def _task_gold(**context) -> dict:
    affected = context["ti"].xcom_pull(task_ids="ge_gate") or []
    if not affected:
        print("[pipeline.gold] no affected partitions; skipping gold build")
        return {"review_dates": [], "total_silver_rows": 0}

    process_review_dates(_SILVER_ROOT, _GOLD_ROOT, affected)

    counts = {
        key: len(read_silver_partition(silver_partition_path(_SILVER_ROOT, key)))
        for key in affected
    }
    total = sum(counts.values())
    print(
        f"[pipeline.gold] {total} silver rows across "
        f"{len(affected)} review_date partition(s)"
    )
    return {"review_dates": affected, "silver_row_counts": counts, "total_silver_rows": total}


def _task_publish(**context) -> dict:
    info = context["ti"].xcom_pull(task_ids="gold") or {}
    affected = info.get("review_dates") or []
    if not affected:
        print("[pipeline.publish] no affected partitions; nothing to publish")
        return {}
    from data.publish import publish_run 

    run_date = context["ds"]
    summary = publish_run(
        sorted(affected),
        run_date,
        bronze_root=_BRONZE_ROOT,
        silver_root=_SILVER_ROOT,
        gold_root=_GOLD_ROOT,
    )
    print(f"[pipeline.publish] {summary}")
    return summary or {}


def _should_train(**context) -> bool:
    forced = os.getenv("FORCE_TRAIN") == "1"
    print(f"[pipeline.should_train] FORCE_TRAIN={'1' if forced else '0'} -> train={forced}")
    return forced


def _task_train(**_context) -> dict:
    csv_path = materialize_training_csv(_TRAINING_CSV, _GOLD_ROOT)
    result = train_run(data_path=csv_path)
    print(
        f"[pipeline.train] f1_macro={result.f1_macro:.3f} f1_neg={result.f1_neg:.3f} "
        f"n_train={result.n_train} -> {result.artifact_path} "
        f"(mlflow_version={result.mlflow_model_version})"
    )
    return asdict(result)


def _task_gate(**context) -> dict:
    import pickle

    import pandas as pd
    import psycopg2

    from data.refine.build_gold import label_from_rating
    from monitoring.drift_checks import (
        DRIFT_RECENT_PARTITIONS,
        _features_from_reviews,
        _load_recent_silver,
        evaluate,
    )

    train_res = context["ti"].xcom_pull(task_ids="train") or {}
    artifact_path = train_res.get("artifact_path")
    if not artifact_path or not Path(artifact_path).exists():
        print("[pipeline.gate] no model artifact found; skipping gate (promotion ungated)")
        return {"blocked_promotion": False, "skipped": True}

    with open(artifact_path, "rb") as fh:
        model = pickle.load(fh)

    # reference = the training frame (text + label); current = recent silver
    reference = pd.read_csv(_TRAINING_CSV)
    recent = _load_recent_silver(_SILVER_ROOT, DRIFT_RECENT_PARTITIONS)
    if recent is None or recent.empty:
        print("[pipeline.gate] no recent silver window; gating reference-vs-itself")
        current = reference.copy()
    else:
        current = _features_from_reviews(recent)
        if "rating" in current.columns:
            current["label"] = [
                label_from_rating(r) if pd.notna(r) else None
                for r in current["rating"]
            ]
        current = current.dropna(subset=["label"])

    conn = psycopg2.connect(_app_dsn())
    try:
        with conn:
            result = evaluate(
                reference,
                current,
                conn,
                _minio_client(),
                run_date=date.fromisoformat(context["ds"]),
                model=model,
                report_type="performance",
                raise_on_block=False,
            )
    finally:
        conn.close()

    print(
        f"[pipeline.gate] blocked_promotion={result['blocked_promotion']} "
        f"drift_score={result['drift_score']:.3f} f1_drop={result['f1_drop']} "
        f"recall_neg_drop={result['recall_neg_drop']} -> {result['report_url']}"
    )
    return result


def _task_promote(**context) -> bool:
    gate = context["ti"].xcom_pull(task_ids="gate") or {}
    if gate.get("blocked_promotion"):
        print(
            "[pipeline.promote] gate blocked promotion "
            f"(drift_score={gate.get('drift_score')}, f1_drop={gate.get('f1_drop')}, "
            f"recall_neg_drop={gate.get('recall_neg_drop')}); leaving model unstaged"
        )
        return False

    result = context["ti"].xcom_pull(task_ids="train") or {}
    promoted = promote_to_production(
        version=result.get("mlflow_model_version"),
        metrics=result,
    )
    print(f"[pipeline.promote] promoted={promoted}")
    return promoted


def _task_reload_api(**context) -> None:
    promoted = context["ti"].xcom_pull(task_ids="promote")
    if not promoted:
        print("[pipeline.reload_api] nothing promoted; skipping API reload")
        return

    api_url = os.getenv("API_URL", "http://api:8000")
    token = os.getenv("ADMIN_TOKEN")
    if not token:
        print("[pipeline.reload_api] ADMIN_TOKEN unset; skipping API reload")
        return

    import urllib.error
    import urllib.request

    req = urllib.request.Request(
        f"{api_url}/reload",
        method="POST",
        headers={"Authorization": f"Bearer {token}"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted internal URL)
            print(f"[pipeline.reload_api] {api_url}/reload -> {resp.status} {resp.read().decode()}")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[pipeline.reload_api] reload failed ({exc}); model still promoted in MLflow")


with DAG(
    dag_id="medallion_pipeline",
    description="6h data refresh (bronze→silver→GE→gold→publish) + on-demand (FORCE_TRAIN=1) train→gate→promote→reload",
    start_date=datetime(2025, 1, 1),
    schedule="0 */6 * * *",
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["pipeline", "medallion", "model"],
) as dag:
    dep_check_source_data = EmptyOperator(task_id="dep_check_source_data")
    bronze = PythonOperator(task_id="bronze", python_callable=_task_bronze)
    silver = PythonOperator(task_id="silver", python_callable=_task_silver)
    ge_gate = PythonOperator(task_id="ge_gate", python_callable=_task_ge_gate)
    gold = PythonOperator(task_id="gold", python_callable=_task_gold)
    # outlets: emit the dataset event on success -> data-triggers batch_inference.
    publish = PythonOperator(
        task_id="publish",
        python_callable=_task_publish,
        outlets=[REVIEWS_GOLD_DATASET],
    )

    should_train = ShortCircuitOperator(
        task_id="should_train",
        python_callable=_should_train,
        ignore_downstream_trigger_rules=False,
    )
    train = PythonOperator(task_id="train", python_callable=_task_train)
    gate = PythonOperator(task_id="gate", python_callable=_task_gate)
    promote = PythonOperator(task_id="promote", python_callable=_task_promote)
    reload_api = PythonOperator(task_id="reload_api", python_callable=_task_reload_api)

    pipeline_completed = EmptyOperator(
        task_id="pipeline_completed",
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    dep_check_source_data >> bronze >> silver >> ge_gate >> gold >> publish

    gold >> should_train >> train >> gate >> promote >> reload_api

    [publish, reload_api] >> pipeline_completed
