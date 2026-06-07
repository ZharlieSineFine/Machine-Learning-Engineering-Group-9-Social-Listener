"""Training entry point — callable from CLI and from the Airflow DAG.

Reads the sample CSV, fits `baseline_sklearn`, writes a pickle artifact, and
(optionally) logs the run to MLflow. The smoke test calls `run()` directly;
the Airflow DAG calls the same function via PythonOperator.

CLI:
    python models/train.py                       # uses defaults
    python models/train.py --data data/sample/reviews_sample.csv \\
                           --out models/artifacts/baseline.pkl

MLflow is best-effort: if MLFLOW_TRACKING_URI is unset or the server is down,
training still succeeds and the pickle is still written. This is what keeps
the smoke test runnable without the full stack online.

Owner: Van (Modeler).
"""
from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.baseline_sklearn import train  # noqa: E402

DEFAULT_DATA = ROOT / "data" / "sample" / "reviews_sample.csv"
DEFAULT_OUT = ROOT / "models" / "artifacts" / "baseline.pkl"



@dataclass
class TrainResult:
    artifact_path: str
    f1_macro: float        # in-time test
    f1_weighted: float     # in-time test
    n_train: int
    n_test: int
    n_val: int = 0
    n_oot: int = 0
    f1_macro_oot: Optional[float] = None    # out-of-time (None when no dated rows)
    f1_weighted_oot: Optional[float] = None
    cutoff_date: Optional[str] = None       # first OOT date
    mlflow_run_id: Optional[str] = None


def _try_log_mlflow(metrics: dict, artifact_path: Path) -> Optional[str]:
    """Best-effort MLflow logging. Returns run_id on success, else None."""
    tracking_uri = os.getenv("MLFLOW_TRACKING_URI")
    if not tracking_uri:
        return None
    try:
        import mlflow
        import mlflow.sklearn  # noqa: F401

        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT", "sentiment-baseline"))
        with mlflow.start_run() as run:
            mlflow.log_metric("f1_macro", metrics["f1_macro"])
            mlflow.log_metric("f1_weighted", metrics["f1_weighted"])
            # Out-of-time scores — the test-vs-OOT gap is the temporal-drift signal.
            if metrics.get("f1_macro_oot") is not None:
                mlflow.log_metric("f1_macro_oot", metrics["f1_macro_oot"])
                mlflow.log_metric("f1_weighted_oot", metrics["f1_weighted_oot"])
            for key in ("n_train", "n_val", "n_test", "n_oot"):
                if key in metrics:
                    mlflow.log_param(key, metrics[key])
            if metrics.get("cutoff_date") is not None:
                mlflow.log_param("oot_cutoff_date", metrics["cutoff_date"])
            mlflow.log_artifact(str(artifact_path), artifact_path="model")
            # TODO (member): register the model to the MLflow Model Registry
            # once F1 clears the promotion threshold. The API loads from
            # `models:/sentiment-baseline/Production`.
            return run.info.run_id
    except Exception as exc:  # MLflow optional in smoke test
        print(f"[train] MLflow logging skipped: {exc}")
        return None


def run(data_path: Path = DEFAULT_DATA, out_path: Path = DEFAULT_OUT) -> TrainResult:
    df = pd.read_csv(data_path)
    pipe, metrics = train(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(pipe, f)

    run_id = _try_log_mlflow(metrics, out_path)

    return TrainResult(
        artifact_path=str(out_path),
        f1_macro=metrics["f1_macro"],
        f1_weighted=metrics["f1_weighted"],
        n_train=metrics["n_train"],
        n_test=metrics["n_test"],
        n_val=metrics.get("n_val", 0),
        n_oot=metrics.get("n_oot", 0),
        f1_macro_oot=metrics.get("f1_macro_oot"),
        f1_weighted_oot=metrics.get("f1_weighted_oot"),
        cutoff_date=metrics.get("cutoff_date"),
        mlflow_run_id=run_id,
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
