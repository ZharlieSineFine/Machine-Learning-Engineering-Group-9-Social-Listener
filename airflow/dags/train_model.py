"""Airflow DAG — every-6h retrain of the logreg sentiment baseline.

Thin wrapper around ``models.train.run``. All real logic (fit, pickle, MLflow
log + register) lives in the pure module so it stays unit-testable without
Airflow, like the other DAGs.

MLflow env (MLFLOW_TRACKING_URI, MODEL_NAME, MLFLOW_EXPERIMENT,
MLFLOW_S3_ENDPOINT_URL, AWS_*) is injected into the Airflow containers by
docker-compose — the same block evaluate_and_monitor relies on. When
MLFLOW_TRACKING_URI is set, logging is required (a failed log fails the run);
when unset, only the pickle is produced.

Phase 2: point ``data_path`` at the gold feature/label store once a
training-frame assembler exists. For now it uses the in-repo sample CSV
(models/train.py default) so the DAG is runnable immediately.

Owner: Van (Modeler).
"""
from __future__ import annotations

import sys
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path

_REPO_ROOT = Path("/opt/project")
if _REPO_ROOT.exists() and str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from airflow import DAG
from airflow.operators.python import PythonOperator

from models.train import run as train_run


def _task_train(**_context) -> dict:
    result = train_run()  # defaults: sample CSV -> models/artifacts/baseline.pkl
    print(
        f"[train_model] f1_macro={result.f1_macro:.3f} f1_neg={result.f1_neg:.3f} "
        f"n_train={result.n_train} -> {result.artifact_path} "
        f"(mlflow_version={result.mlflow_model_version})"
    )
    return asdict(result)


with DAG(
    dag_id="train_model",
    description="Manual-only train + register utility (medallion_train_cycle is the scheduled retrain)",
    start_date=datetime(2025, 1, 1),
    # Manual/trigger-only. The canonical scheduled retrain is medallion_train_cycle
    # (weekly + drift-triggered); this DAG stays for ad-hoc train+register runs.
    schedule=None,
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["model", "training"],
) as dag:
    PythonOperator(
        task_id="train_baseline",
        python_callable=_task_train,
    )
