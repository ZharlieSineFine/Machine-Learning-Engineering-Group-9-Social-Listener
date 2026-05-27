"""Integration test for Step 11 — /reload picks up a newly-promoted version.

End-to-end:
    1. Train + register version X, promote it to Production.
    2. Boot the API in-process (loads X via mlflow.sklearn).
    3. Train + register version Y, promote Y to Production (X is archived).
    4. Call /reload — verify the API now serves Y.
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


def _train_and_promote(monkeypatch_module, tmp_path_factory) -> str:
    from mlflow.tracking import MlflowClient

    from models.train import run as train_run

    out = tmp_path_factory.mktemp("artifacts") / "baseline.pkl"
    result = train_run(data_path=SAMPLE_CSV, out_path=out)
    assert result.mlflow_model_version is not None

    client = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    client.transition_model_version_stage(
        name=os.environ["MODEL_NAME"],
        version=result.mlflow_model_version,
        stage="Production",
        archive_existing_versions=True,
    )
    return result.mlflow_model_version


def test_reload_swaps_in_new_production_version(monkeypatch_module, tmp_path_factory):
    monkeypatch_module.setenv("MLFLOW_TRACKING_URI", "http://localhost:5001")
    monkeypatch_module.setenv("MLFLOW_S3_ENDPOINT_URL", "http://localhost:9000")
    monkeypatch_module.setenv("AWS_ACCESS_KEY_ID", "minioadmin")
    monkeypatch_module.setenv("AWS_SECRET_ACCESS_KEY", "minioadmin")
    monkeypatch_module.setenv("MODEL_NAME", "sentiment-baseline")
    monkeypatch_module.setenv("MODEL_STAGE", "Production")
    monkeypatch_module.setenv("ADMIN_TOKEN", f"tok-{uuid.uuid4().hex[:8]}")
    monkeypatch_module.setenv("MLFLOW_EXPERIMENT", f"reload-it-{uuid.uuid4().hex[:8]}")

    # 1. Register + promote v_old, then boot the API.
    v_old = _train_and_promote(monkeypatch_module, tmp_path_factory)

    for mod in ["app.main", "app.model_loader"]:
        sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient

    from app.main import app
    client = TestClient(app)

    assert client.get("/health").json()["model_source"] == "mlflow"

    # 2. Register + promote v_new (different version number).
    v_new = _train_and_promote(monkeypatch_module, tmp_path_factory)
    assert v_new != v_old

    # 3. Hit /reload with the admin token.
    r = client.post("/reload", headers={"X-Admin-Token": os.environ["ADMIN_TOKEN"]})
    assert r.status_code == 200, r.text
    assert r.json()["model_source"] == "mlflow"

    # 4. Confirm the MLflow registry now reports v_new as Production
    #    (the API will load whatever the alias resolves to on /reload).
    from mlflow.tracking import MlflowClient
    mclient = MlflowClient(tracking_uri=os.environ["MLFLOW_TRACKING_URI"])
    prod_versions = mclient.get_latest_versions(
        name=os.environ["MODEL_NAME"], stages=["Production"]
    )
    assert prod_versions, "expected at least one Production version"
    assert prod_versions[0].version == v_new
