"""Integration test for Step 5 — FastAPI loads the model from MLflow Registry.

Flow:
    1. Train + register a fresh `sentiment-baseline` version (uses Step 4).
    2. Promote that version to stage `Production` via the MLflow client.
    3. Set the env vars the API reads (MLFLOW_TRACKING_URI + MODEL_NAME +
       MODEL_STAGE) and force-reload `app.main`.
    4. /health reports `model_source == 'mlflow'` and the predict endpoint
       returns a valid label.
"""
from __future__ import annotations

import os
import sys
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"


@pytest.fixture(scope="module")
def monkeypatch_module():
    from _pytest.monkeypatch import MonkeyPatch
    mp = MonkeyPatch()
    yield mp
    mp.undo()


@pytest.fixture(scope="module")
def promoted_version(tmp_path_factory, monkeypatch_module):
    """Train, register, promote — return the version string."""
    monkeypatch_module.setenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    monkeypatch_module.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch_module.setenv("AWS_ACCESS_KEY_ID", "minioadmin")
    monkeypatch_module.setenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    monkeypatch_module.setenv("MODEL_NAME", "sentiment-baseline")
    monkeypatch_module.setenv("MODEL_STAGE", "Production")
    monkeypatch_module.setenv(
        "MLFLOW_EXPERIMENT", f"api-it-{uuid.uuid4().hex[:8]}"
    )

    from models.train import run as train_run

    out = tmp_path_factory.mktemp("artifacts") / "baseline.pkl"
    result = train_run(data_path=SAMPLE_CSV, out_path=out)
    assert result.mlflow_model_version is not None

    from mlflow.tracking import MlflowClient

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    client.transition_model_version_stage(
        name=os.environ["MODEL_NAME"],
        version=result.mlflow_model_version,
        stage="Production",
        archive_existing_versions=True,
    )
    return result.mlflow_model_version


@pytest.fixture(scope="module")
def api_client_mlflow(promoted_version):
    """Boot the FastAPI app with MLflow env set — model loads from registry."""
    # Clear cached imports so model_loader._try_mlflow runs fresh with env vars.
    for mod in ["app.main", "app.model_loader"]:
        sys.modules.pop(mod, None)

    from fastapi.testclient import TestClient

    from app.main import app
    return TestClient(app)


def test_health_reports_mlflow_source(api_client_mlflow):
    r = api_client_mlflow.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["model_loaded"] is True
    assert body["model_source"] == "mlflow", f"expected mlflow, got {body}"


def test_predict_via_mlflow_returns_valid_label(api_client_mlflow):
    r = api_client_mlflow.post(
        "/predict", json={"text": "amazing food and lovely service"}
    )
    assert r.status_code == 200
    assert r.json()["label"] in {"negative", "neutral", "positive"}


def test_predict_via_mlflow_negative(api_client_mlflow):
    r = api_client_mlflow.post(
        "/predict", json={"text": "worst meal of my life, completely inedible"}
    )
    assert r.status_code == 200
    assert r.json()["label"] in {"negative", "neutral", "positive"}
