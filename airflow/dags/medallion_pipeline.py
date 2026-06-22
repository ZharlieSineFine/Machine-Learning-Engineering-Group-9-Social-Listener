"""Airflow DAG — the build pipeline: data every 6h + a weekly model retrain.

Runs every 6h and builds the medallion; the train→promote branch is gated to one
run per week:

    dep_check → bronze → silver → ge_gate → gold → publish
                                    │
                                    └─ (weekend?) → train → gate → promote → reload_api
                                                                      └──────┴─► completed

This consolidates the old ``run_daily_medallion`` (data) and
``medallion_train_cycle`` (retrain) DAGs, which shared the bronze→gold prefix.
Drift **monitoring lives in its own DAG** (``evaluate_and_monitor``) — it's a
read-only observer, kept separate so an Evidently hiccup pages as a *monitoring*
failure, not a pipeline failure. The heavy DistilBERT challenger likewise stays in
``shadow_deploy_distilbert`` (deps that aren't in the Airflow image).

Design choice — **data every 6h, model weekly.** The data layers refresh on every
run; the train→promote branch is short-circuited except on the Sunday 00:00 run
(or when ``FORCE_TRAIN=1`` is set, for demos/off-cycle drift response/tests). One
retrain/week is plenty for the review volume, and the cheap logreg baseline could
go faster if needed. There is no auto-retrain on drift: ``evaluate_and_monitor``
alerts a human, who triggers this DAG with ``FORCE_TRAIN=1`` if warranted.

Every task body is a thin wrapper around the same pure functions the CLIs/tests
use (``data.run_daily`` / ``data.refine`` / ``monitoring.drift_checks`` /
``models.*``) — nothing is reimplemented here.

Owner: Charlie + Ha (Data & Eval) + Van (Modeler).
"""
from __future__ import annotations

import os
import sys
from dataclasses import asdict
from datetime import date, datetime, timedelta
from pathlib import Path

# /opt/project is the in-container mount of the repo root (see docker-compose),
# so `from data...` / `from models...` resolve from inside the DAG.
_REPO_ROOT = Path("/opt/project")
if _REPO_ROOT.exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airflow import DAG
from airflow.operators.empty import EmptyOperator
from airflow.operators.python import PythonOperator, ShortCircuitOperator
from airflow.utils.trigger_rule import TriggerRule

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


# ---------------------------------------------------------------------------
# shared infra helpers (env-gated, same env docker-compose injects)
# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
# data layers: bronze → silver → ge_gate → gold → publish
# ---------------------------------------------------------------------------
def _task_bronze(**context) -> None:
    run_date = context["ds"]  # YYYY-MM-DD from Airflow logical date
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
    return affected  # -> XCom


def _task_ge_gate(**context) -> list[str]:
    affected = context["ti"].xcom_pull(task_ids="silver") or []
    if not affected:
        print("[pipeline.ge_gate] no affected partitions; skipping validation")
        return affected
    validate_silver_partitions(_SILVER_ROOT, affected)  # raises DailyRunError on violation
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
    """Mirror the affected partitions to MinIO + upsert Postgres.

    Env-gated via ``data.publish.publish_run``: a clean no-op when MinIO/Postgres
    aren't configured, so the DAG stays green in environments without them.
    """
    info = context["ti"].xcom_pull(task_ids="gold") or {}
    affected = info.get("review_dates") or []
    if not affected:
        print("[pipeline.publish] no affected partitions; nothing to publish")
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
    print(f"[pipeline.publish] {summary}")
    return summary or {}


# ---------------------------------------------------------------------------
# model branch: (weekend?) → train → gate → promote → reload_api
# ---------------------------------------------------------------------------
def _should_train(**context) -> bool:
    """ShortCircuit: gate the train→promote branch to one run per week.

    Data refreshes every 6h, but the model retrains weekly — on the Sunday 00:00
    run. ``FORCE_TRAIN=1`` overrides for demos, off-cycle drift response, or tests.
    """
    if os.getenv("FORCE_TRAIN") == "1":
        print("[pipeline.should_train] FORCE_TRAIN=1 -> training this run")
        return True
    logical = context["logical_date"]
    # weekday(): Mon=0 .. Sun=6. Runs fire at 00/06/12/18; hour<6 keeps it to the
    # single midnight run so we train once, not four times, on Sunday.
    is_weekly_slot = logical.weekday() == 6 and logical.hour < 6
    print(
        f"[pipeline.should_train] {logical.isoformat()} weekday={logical.weekday()} "
        f"hour={logical.hour} -> train={is_weekly_slot}"
    )
    return is_weekly_slot


def _task_train(**_context) -> dict:
    # Build the training frame from Gold (falls back to sample CSV if Gold empty),
    # then train via the unchanged models.train.run entry point.
    csv_path = materialize_training_csv(_TRAINING_CSV, _GOLD_ROOT)
    result = train_run(data_path=csv_path)
    print(
        f"[pipeline.train] f1_macro={result.f1_macro:.3f} f1_neg={result.f1_neg:.3f} "
        f"n_train={result.n_train} -> {result.artifact_path} "
        f"(mlflow_version={result.mlflow_model_version})"
    )
    return asdict(result)


def _task_gate(**context) -> dict:
    """Promotion gate: data drift OR performance regression between the training
    frame (reference) and the recent silver window (current).

    Loads the just-trained model, scores it on both sides (macro-F1 + negative-class
    recall), runs the Evidently report, uploads HTML + writes the
    ``monitoring_reports`` row, and returns ``blocked_promotion`` for ``promote`` to
    honour. Never raises — promotion is gated by the flag, not by a failed task.
    """
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

    # reference = the training frame (text + label); current = recent silver with
    # labels derived from rating (silver carries no label of its own).
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
        with conn:  # commit on success
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
    # Best-effort: the model is already in Production (durable). A flaky reload
    # shouldn't fail the cycle — the API picks the new model up on its next
    # restart/load regardless. Log and move on.
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 (trusted internal URL)
            print(f"[pipeline.reload_api] {api_url}/reload -> {resp.status} {resp.read().decode()}")
    except (urllib.error.URLError, TimeoutError) as exc:
        print(f"[pipeline.reload_api] reload failed ({exc}); model still promoted in MLflow")


with DAG(
    dag_id="medallion_pipeline",
    description="6h data refresh (bronze→silver→GE→gold→publish) + weekly train→gate→promote→reload",
    start_date=datetime(2025, 1, 1),
    schedule="0 */6 * * *",  # every 6h, per ARCHITECTURE.md §3 batch cycle
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["pipeline", "medallion", "model"],
) as dag:
    # --- data layers (run every 6h) ---
    dep_check_source_data = EmptyOperator(task_id="dep_check_source_data")
    bronze = PythonOperator(task_id="bronze", python_callable=_task_bronze)
    silver = PythonOperator(task_id="silver", python_callable=_task_silver)
    ge_gate = PythonOperator(task_id="ge_gate", python_callable=_task_ge_gate)
    gold = PythonOperator(task_id="gold", python_callable=_task_gold)
    publish = PythonOperator(task_id="publish", python_callable=_task_publish)

    # --- model branch (weekly: Sunday 00:00, or FORCE_TRAIN=1) ---
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
        # Runs once the data branch is done and the optional branches have either
        # finished or been short-circuited (skipped) — never on upstream failure.
        trigger_rule=TriggerRule.NONE_FAILED,
    )

    # data spine
    dep_check_source_data >> bronze >> silver >> ge_gate >> gold >> publish

    # model branch fans out from gold (weekly: trains on the data just built)
    gold >> should_train >> train >> gate >> promote >> reload_api

    # single terminal node the whole graph converges on
    [publish, reload_api] >> pipeline_completed
