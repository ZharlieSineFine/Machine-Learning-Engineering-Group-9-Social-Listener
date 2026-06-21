"""Register the champion baseline in the MLflow registry.

Gives the API a `models:/sentiment-baseline/Production` to serve and a `.../Staging`
candidate for the shadow-deploy panel, and gives the MLOps-monitor page a real
registry to show. Inference itself still loads the local pickle (faster) — this is
the "registry + serving" half of the story.

Logs the champion (champion_baseline_v3.pkl) wrapped as a TunedSentimentPipeline
(string labels + tuned negative threshold) twice:
  * neg_threshold 0.46 -> Production  (logreg-final, the champion)
  * neg_threshold 0.40 -> Staging     (logreg-candidate, the shadow challenger)

Idempotent: skips if sentiment-baseline already has a Production version.

Run (host):       python scripts/register_champion.py
Run (in-stack):   docker exec sentiment-airflow-scheduler python /opt/project/scripts/register_champion.py
"""
from __future__ import annotations

import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

MODEL_NAME = os.getenv("MODEL_NAME", "sentiment-baseline")
TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
EXPERIMENT = os.getenv("MLFLOW_EXPERIMENT", "sentiment-tfidf-logreg")
PICKLE = Path(os.getenv("MODEL_PICKLE_PATH", str(ROOT / "models" / "artifacts" / "baseline.pkl")))

# Champion metrics from models/champion_manifest.txt (logreg-final, registry v3).
CHAMPION_METRICS = {
    "f1_macro": 0.7208,
    "f1_neg": 0.8727,
    "recall_neg": 0.9034,
    "precision_neg": 0.8441,
    "test_f1_macro": 0.7208,
    "test_f1_negative": 0.8727,
    "test_recall_negative": 0.9034,
    "test_precision_negative": 0.8441,
    "oot_f1_negative": 0.8635,
}


def main() -> None:
    # MinIO creds for artifact upload (no-ops if already set by the container env).
    os.environ.setdefault("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    os.environ.setdefault("AWS_ACCESS_KEY_ID", "minioadmin")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "minioadmin")

    import mlflow
    import mlflow.sklearn
    from mlflow.tracking import MlflowClient

    from models.baseline_sklearn import TunedSentimentPipeline

    mlflow.set_tracking_uri(TRACKING_URI)
    client = MlflowClient(TRACKING_URI)

    # Idempotent: skip if a Production version already exists.
    try:
        versions = client.search_model_versions(f"name='{MODEL_NAME}'")
        if any(v.current_stage == "Production" for v in versions):
            prod = next(v for v in versions if v.current_stage == "Production")
            print(f"[register] {MODEL_NAME} already has Production v{prod.version}; skipping.")
            return
    except Exception:
        pass  # registered model doesn't exist yet — fall through and create it

    with open(PICKLE, "rb") as fh:
        raw_pipeline = pickle.load(fh)

    mlflow.set_experiment(EXPERIMENT)
    code_paths = [str(ROOT / "models")]

    def register(neg_threshold: float, stage: str, run_name: str) -> None:
        model = TunedSentimentPipeline(pipeline=raw_pipeline, neg_threshold=neg_threshold)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_param("model_type", "tfidf_logreg_baseline")
            mlflow.log_param("neg_threshold", neg_threshold)
            mlflow.log_metrics(CHAMPION_METRICS)
            info = mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path="model",
                registered_model_name=MODEL_NAME,
                code_paths=code_paths,
            )
        version = getattr(info, "registered_model_version", None)
        client.transition_model_version_stage(
            name=MODEL_NAME, version=str(version), stage=stage,
            archive_existing_versions=False,
        )
        print(f"[register] {MODEL_NAME} v{version} -> {stage} (neg_threshold={neg_threshold}, run={run_name})")

    register(0.46, "Production", "logreg-final")
    register(0.40, "Staging", "logreg-candidate")
    print("[register] done.")


if __name__ == "__main__":
    main()
