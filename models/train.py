"""Training entry point — callable from CLI and from the Airflow DAG.

Reads the sample CSV, fits `baseline_sklearn`, writes a pickle artifact, and
logs the run + registers the model to MLflow.

CLI:
    python models/train.py                       # uses defaults
    python models/train.py --data data/sample/reviews_sample.csv \\
                           --out models/artifacts/baseline.pkl

MLflow behavior:
    - If MLFLOW_TRACKING_URI is set, logging is REQUIRED. Failure to log
      fails the training run, by design — silent skips would let bad runs
      ship to production.
    - If MLFLOW_TRACKING_URI is unset (smoke-test / offline mode), MLflow
      is skipped entirely and only the pickle is produced.

The pickle remains as a fallback for the FastAPI service when the registry
isn't reachable. See api/app/model_loader.py.

Owner: Van (Modeler).
"""
from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baseline_sklearn import train  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "sample" / "reviews_sample.csv"
DEFAULT_OUT = ROOT / "models" / "artifacts" / "baseline.pkl"
DEFAULT_MODEL_NAME = "sentiment-baseline"
DEFAULT_EXPERIMENT = "sentiment-baseline"


@dataclass
class TrainResult:
    artifact_path: str
    f1_macro: float
    f1_weighted: float
    accuracy: float
    f1_neg: float
    precision_neg: float
    recall_neg: float
    n_train: int
    n_test: int
    n_val: int = 0
    n_oot: int = 0
    f1_macro_oot: Optional[float] = None
    f1_weighted_oot: Optional[float] = None
    cutoff_date: Optional[str] = None
    mlflow_run_id: Optional[str] = None
    mlflow_model_version: Optional[str] = None


def _log_to_mlflow(pipe: Any, metrics: dict) -> tuple[str, Optional[str]]:
    """Log run + register model. Returns (run_id, model_version).

    Raises if MLflow is unreachable — by design. The caller has already
    decided we *should* log (by checking MLFLOW_TRACKING_URI).
    """
    import mlflow
    import mlflow.sklearn

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    experiment = os.getenv("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as run:
        tuned = metrics.get("tuned_params", {})
        mlflow.log_params({
            "model_type": "tfidf_logreg_baseline",
            "n_train": metrics["n_train"],
            "n_val": metrics.get("n_val", 0),
            "n_test": metrics["n_test"],
            "neg_threshold": metrics.get("neg_threshold", ""),
            "tfidf__ngram_range": str(tuned.get("tfidf__ngram_range", "")),
            "tfidf__max_features": tuned.get("tfidf__max_features", ""),
            "clf__C": tuned.get("clf__C", ""),
            "clf__class_weight": str(tuned.get("clf__class_weight", "")),
        })
        if metrics.get("cutoff_date") is not None:
            mlflow.log_param("oot_cutoff_date", metrics["cutoff_date"])

        logged_metrics = {
            "f1_macro": metrics["f1_macro"],
            "f1_weighted": metrics["f1_weighted"],
            "accuracy": metrics["accuracy"],
            "f1_neg": metrics["f1_neg"],
            "precision_neg": metrics["precision_neg"],
            "recall_neg": metrics["recall_neg"],
        }
        if metrics.get("f1_macro_oot") is not None:
            logged_metrics["f1_macro_oot"] = metrics["f1_macro_oot"]
            logged_metrics["f1_weighted_oot"] = metrics["f1_weighted_oot"]
        mlflow.log_metrics(logged_metrics)

        info = mlflow.sklearn.log_model(
            sk_model=pipe,
            artifact_path="model",
            registered_model_name=model_name,
            code_paths=[str(ROOT / "models")],
        )
        version = getattr(info, "registered_model_version", None)
        return run.info.run_id, str(version) if version else None


def run(data_path: Path = DEFAULT_DATA, out_path: Path = DEFAULT_OUT) -> TrainResult:
    df = pd.read_csv(data_path)
    pipe, metrics = train(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)

    run_id: Optional[str] = None
    version: Optional[str] = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        run_id, version = _log_to_mlflow(pipe, metrics)

    return TrainResult(
        artifact_path=str(out_path),
        f1_macro=metrics["f1_macro"],
        f1_weighted=metrics["f1_weighted"],
        accuracy=metrics["accuracy"],
        f1_neg=metrics["f1_neg"],
        precision_neg=metrics["precision_neg"],
        recall_neg=metrics["recall_neg"],
        n_train=metrics["n_train"],
        n_test=metrics["n_test"],
        n_val=metrics.get("n_val", 0),
        n_oot=metrics.get("n_oot", 0),
        f1_macro_oot=metrics.get("f1_macro_oot"),
        f1_weighted_oot=metrics.get("f1_weighted_oot"),
        cutoff_date=metrics.get("cutoff_date"),
        mlflow_run_id=run_id,
        mlflow_model_version=version,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=DEFAULT_DATA)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    result = run(args.data, args.out)
    for k, v in asdict(result).items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
