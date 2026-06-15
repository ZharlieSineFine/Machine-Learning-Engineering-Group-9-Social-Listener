"""End-to-end smoke test for Phase 1.

Walks the full thin-slice without booting Docker:
    1. Sample CSV exists, has the contract columns.
    2. Baseline pipeline fits and predicts on the sample.
    3. `train.run()` writes a pickle to disk.
    4. FastAPI loads that pickle and /health + /predict respond correctly.

This is intentionally fast (<10s on a laptop) and depends on NO services
(no Postgres, MinIO, MLflow, Airflow). Those get their own integration
tests once Phase 2 wiring lands.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from models import train as train_mod
from models.baseline_sklearn import LABELS, train

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"

EXPECTED_COLUMNS = {"text", "label", "rating", "source", "restaurant", "location"}


def test_sample_csv_contract():
    assert SAMPLE_CSV.exists(), f"missing sample: {SAMPLE_CSV}"
    df = pd.read_csv(SAMPLE_CSV)
    assert len(df) > 100, "sample too small to train"
    assert EXPECTED_COLUMNS.issubset(df.columns), f"missing columns: {EXPECTED_COLUMNS - set(df.columns)}"
    assert set(df["label"].unique()).issubset(set(LABELS))


def test_baseline_trains_and_predicts():
    df = pd.read_csv(SAMPLE_CSV)
    pipe, metrics = train(df)

    # Sanity bounds — not a quality gate, just "model learned something".
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert 0.0 <= metrics["recall_neg"] <= 1.0

    preds = pipe.predict(["the food was amazing", "absolutely terrible service"])
    assert all(p in LABELS for p in preds)


def test_train_run_writes_artifact(tmp_path: Path):
    out = tmp_path / "baseline.pkl"
    result = train_mod.run(data_path=SAMPLE_CSV, out_path=out)
    assert out.exists()
    assert result.f1_macro > 0
    assert result.recall_neg > 0
    assert result.mlflow_run_id is None  # no MLflow in smoke test


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    """Build the API with a fresh pickle so it doesn't depend on prior runs."""
    artifact = tmp_path_factory.mktemp("artifacts") / "baseline.pkl"
    train_mod.run(data_path=SAMPLE_CSV, out_path=artifact)

    import os
    os.environ["MODEL_PICKLE_PATH"] = str(artifact)
    # Drop a cached import so model_loader re-reads MODEL_PICKLE_PATH.
    import sys
    for mod in ["app.main", "app.model_loader"]:
        sys.modules.pop(mod, None)

    from app.main import app  # noqa: WPS433 — deliberate late import
    return TestClient(app)


def test_health(api_client):
    r = api_client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True
    assert body["model_source"] == "pickle"


def test_predict_positive_and_negative(api_client):
    pos = api_client.post("/predict", json={"text": "Amazing food, friendly staff, would return"})
    neg = api_client.post("/predict", json={"text": "Worst meal of my life, completely inedible"})
    assert pos.status_code == 200
    assert neg.status_code == 200
    assert pos.json()["label"] in LABELS
    assert neg.json()["label"] in LABELS


def test_predict_rejects_empty(api_client):
    r = api_client.post("/predict", json={"text": ""})
    assert r.status_code == 422  # Pydantic min_length=1
