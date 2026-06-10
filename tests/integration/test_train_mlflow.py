"""Integration test for Step 4 — train.py against a real MLflow server.

Verifies:
  1. `run()` returns a non-empty mlflow_run_id and model_version.
  2. The run appears in MLflow with the expected metrics + params.
  3. A new version of `sentiment-baseline` is registered.
"""
from __future__ import annotations

import os
import uuid
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"


@pytest.fixture(scope="module")
def mlflow_env(monkeypatch_module):
    """Set the env vars the MLflow client + S3 artifact upload need.

    These differ from the in-container `.env` values because the test runs
    on the host:
        - MLFLOW_TRACKING_URI    -> localhost:5001
        - MLFLOW_S3_ENDPOINT_URL -> localhost:9000 (MinIO from host)
    """
    monkeypatch_module.setenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    monkeypatch_module.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch_module.setenv("AWS_ACCESS_KEY_ID", "minioadmin")
    monkeypatch_module.setenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    # Unique experiment per run so concurrent test runs don't collide.
    exp = f"it-baseline-{uuid.uuid4().hex[:8]}"
    monkeypatch_module.setenv("MLFLOW_EXPERIMENT", exp)
    monkeypatch_module.setenv("MODEL_NAME", "sentiment-baseline")
    return {"experiment": exp}


@pytest.fixture(scope="module")
def monkeypatch_module():
    """A module-scoped MonkeyPatch (the built-in `monkeypatch` is function-scoped)."""
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


def test_train_logs_run_and_registers_model(tmp_path, mlflow_env):
    from models.train import run

    out = tmp_path / "baseline.pkl"
    result = run(data_path=SAMPLE_CSV, out_path=out)

    assert result.mlflow_run_id, "train.run() should return an MLflow run id"
    assert result.mlflow_model_version is not None
    assert int(result.mlflow_model_version) >= 1

    import mlflow
    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])

    # The run exists and has the expected metrics.
    fetched = client.get_run(result.mlflow_run_id)
    assert "f1_macro" in fetched.data.metrics
    assert "f1_weighted" in fetched.data.metrics
    assert 0.0 <= fetched.data.metrics["f1_macro"] <= 1.0
    assert fetched.data.params["model_type"] == "tfidf_logreg_baseline"

    # The model was registered.
    name = os.environ["MODEL_NAME"]
    versions = client.search_model_versions(f"name='{name}'")
    assert any(v.version == result.mlflow_model_version for v in versions)
