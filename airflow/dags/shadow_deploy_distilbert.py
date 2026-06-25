"""Airflow DAG — DistilBERT shadow deploy (challenger → MLflow Staging).

Fine-tunes the DistilBERT challenger and registers it to MLflow under a
*separate* model name (``sentiment-distilbert``) at the **Staging** stage. It
never reaches Production, so the FastAPI service — which loads only
``models:/{MODEL_NAME}/Production`` (default ``sentiment-baseline``) via
``mlflow.sklearn`` — is unaffected. The challenger simply sits in the registry
ready for a future serving-side shadow lane (ARCHITECTURE.md §4).

Thin wrapper around the unchanged pure functions:
    * ``models.gold_loader.materialize_training_csv`` — Gold → training CSV
      (same helper full_cycle.py uses), with sample-CSV fallback.
    * ``models.distilbert_finetune.train_distilbert`` — fine-tune + save.

Lean-image contract: ``torch`` / ``transformers`` / ``datasets`` are NOT in the
Airflow image. When they're absent the train task raises ``AirflowSkipException``
so the task is **skipped** (not failed) and the DAG list stays green. Wire the
deps into the image to actually train.

Env knobs (all optional):
    DISTILBERT_MODEL_NAME   registered model name      (default sentiment-distilbert)
    DISTILBERT_EXPERIMENT   MLflow experiment          (default sentiment-distilbert)
    DISTILBERT_MAX_STEPS    cap training steps for CPU (default 50; <=0 = full run)
    DISTILBERT_EPOCHS       num train epochs           (default 4)

Owner: Van (Modeler) + Charlie + Ha.
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
from airflow.exceptions import AirflowSkipException
from airflow.operators.python import PythonOperator

_GOLD_ROOT = _REPO_ROOT / "data" / "gold"
_TRAINING_CSV = _GOLD_ROOT / "_distilbert_training_frame.csv"
_OUT_DIR = _REPO_ROOT / "models" / "artifacts" / "distilbert_shadow"

DEFAULT_MODEL_NAME = "sentiment-distilbert"
DEFAULT_EXPERIMENT = "sentiment-distilbert"
STAGING_STAGE = "Staging"


def _require_distilbert_deps() -> None:
    """Skip the task cleanly when the heavy DL deps aren't in the image."""
    try:
        import datasets  # noqa: F401
        import torch  # noqa: F401
        import transformers  # noqa: F401
    except ImportError as exc:
        raise AirflowSkipException(
            "DistilBERT deps (torch/transformers/datasets) not installed; "
            f"skipping shadow train ({exc})"
        )


def _register_staging(out_dir: Path, metrics: dict) -> tuple[str, str | None]:
    """Log the saved model+tokenizer and transition the new version to Staging.

    Returns (model_name, version). No-ops (returns version=None) when
    MLFLOW_TRACKING_URI is unset, matching the other DAGs' offline contract.
    """
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    model_name = os.getenv("DISTILBERT_MODEL_NAME", DEFAULT_MODEL_NAME)
    if not tracking_uri:
        print("[distilbert_shadow] MLFLOW_TRACKING_URI unset — left local artifact only")
        return model_name, None

    import mlflow
    import mlflow.transformers
    from mlflow.tracking import MlflowClient
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    experiment = os.getenv("DISTILBERT_EXPERIMENT", DEFAULT_EXPERIMENT)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    model = AutoModelForSequenceClassification.from_pretrained(str(out_dir))
    tokenizer = AutoTokenizer.from_pretrained(str(out_dir))

    with mlflow.start_run() as run:
        mlflow.log_params({
            "model_type": "distilbert_finetune",
            "base_model": "distilbert-base-uncased",
            "max_steps": os.getenv("DISTILBERT_MAX_STEPS", "50"),
            "num_epochs": os.getenv("DISTILBERT_EPOCHS", "4"),
        })
        loggable = {
            k: float(v)
            for k, v in metrics.items()
            if isinstance(v, (int, float))
        }
        if loggable:
            mlflow.log_metrics(loggable)

        info = mlflow.transformers.log_model(
            transformers_model={"model": model, "tokenizer": tokenizer},
            artifact_path="model",
            registered_model_name=model_name,
            task="text-classification",
        )
        version = getattr(info, "registered_model_version", None)
        print(f"[distilbert_shadow] logged run {run.info.run_id}, version={version}")

    if not version:
        print("[distilbert_shadow] no registered version returned — skipping stage move")
        return model_name, None

    client = MlflowClient(tracking_uri=tracking_uri)
    client.transition_model_version_stage(
        name=model_name,
        version=str(version),
        stage=STAGING_STAGE,
        archive_existing_versions=True,
    )
    print(f"[distilbert_shadow] {model_name} v{version} -> {STAGING_STAGE} (shadow)")
    return model_name, str(version)


def _task_train_and_register(**_context) -> dict:
    _require_distilbert_deps()

    import pandas as pd

    from models.distilbert_finetune import TrainConfig, train_distilbert
    from models.gold_loader import materialize_training_csv

    # Build the training frame from Gold (falls back to sample CSV when empty),
    # reusing the same helper as full_cycle.py:_task_train.
    csv_path = materialize_training_csv(_TRAINING_CSV, _GOLD_ROOT)
    df = pd.read_csv(csv_path)

    cfg = TrainConfig(
        num_epochs=int(os.getenv("DISTILBERT_EPOCHS", "4")),
        max_steps=int(os.getenv("DISTILBERT_MAX_STEPS", "50")),  # cap for CPU runs
    )
    out_dir, metrics = train_distilbert(df, _OUT_DIR, cfg)
    print(
        f"[distilbert_shadow] trained -> {out_dir} "
        f"f1_macro={metrics.get('f1_macro')} f1_neg={metrics.get('f1_neg')}"
    )

    model_name, version = _register_staging(Path(out_dir), metrics)
    return {
        "model_name": model_name,
        "version": version,
        "stage": STAGING_STAGE if version else None,
        "f1_macro": metrics.get("f1_macro"),
        "f1_neg": metrics.get("f1_neg"),
        "accuracy": metrics.get("accuracy"),
        "out_dir": str(out_dir),
    }


with DAG(
    dag_id="shadow_deploy_distilbert",
    description="Fine-tune DistilBERT challenger -> register MLflow Staging (shadow)",
    start_date=datetime(2025, 1, 1),
    # Challenger retraining is human-triggered (manual), like the baseline retrain in
    # medallion_pipeline (FORCE_TRAIN=1) — the model is not retrained on a schedule.
    schedule=None,  # manual / triggered only
    catchup=False,
    default_args={
        "owner": "data",
        "retries": 1,
        "retry_delay": timedelta(minutes=10),
    },
    tags=["model", "shadow", "distilbert", "phase2"],
) as dag:
    PythonOperator(
        task_id="train_and_register_shadow",
        python_callable=_task_train_and_register,
    )
