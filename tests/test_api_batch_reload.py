#Unit/smoke tests for /predict/batch + /reload — no MLflow needed.
#The API loads the local pickle so these tests don't need the compose stack running.

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from models import train as train_mod
from models.baseline_sklearn import LABELS

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    #Train a fresh pickle, point the loader at it, boot the API in-process.
    artifact = tmp_path_factory.mktemp("artifacts") / "baseline.pkl"
    # Train without MLflow so the pickle is the source of truth.
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    train_mod.run(data_path=SAMPLE_CSV, out_path=artifact)
    os.environ["MODEL_PICKLE_PATH"] = str(artifact)

    # Drop cached imports so model_loader picks up the new env.
    for mod in ["app.main", "app.model_loader"]:
        sys.modules.pop(mod, None)
    from app.main import app
    return TestClient(app)

def test_batch_returns_one_label_per_text(api_client):
    r = api_client.post("/predict/batch", json={"texts": [
        "the food was incredible",
        "absolutely awful service",
        "it was alright nothing special",
    ]})
    assert r.status_code == 200, r.text
    body = r.json()
    assert len(body["labels"]) == 3
    assert all(lbl in LABELS for lbl in body["labels"])


def test_batch_rejects_empty_list(api_client):
    r = api_client.post("/predict/batch", json={"texts": []})
    assert r.status_code == 422  # min_length=1


def test_batch_rejects_oversize_list(api_client):
    from app.schemas import MAX_BATCH_SIZE
    r = api_client.post("/predict/batch", json={"texts": ["x"] * (MAX_BATCH_SIZE + 1)})
    assert r.status_code == 422


def test_batch_preserves_order(api_client):
    """Position i of the response corresponds to position i of the request."""
    texts = [
        "amazing food and lovely staff",   # likely positive
        "worst meal ever do not recommend", # likely negative
        "the food was incredible",          # likely positive
    ]
    r = api_client.post("/predict/batch", json={"texts": texts})
    labels = r.json()["labels"]
    assert len(labels) == len(texts)
    assert all(isinstance(lbl, str) for lbl in labels)


def test_reload_disabled_when_admin_token_unset(api_client):
    os.environ.pop("ADMIN_TOKEN", None)
    r = api_client.post("/reload")
    assert r.status_code == 503
    assert "disabled" in r.json()["detail"].lower()


def test_reload_requires_token(api_client):
    os.environ["ADMIN_TOKEN"] = "let-me-in"
    try:
        r = api_client.post("/reload")  # no header
        assert r.status_code == 401
    finally:
        os.environ.pop("ADMIN_TOKEN", None)


def test_reload_rejects_wrong_token(api_client):
    os.environ["ADMIN_TOKEN"] = "let-me-in"
    try:
        r = api_client.post("/reload", headers={"X-Admin-Token": "nope"})
        assert r.status_code == 401
    finally:
        os.environ.pop("ADMIN_TOKEN", None)


def test_reload_succeeds_with_correct_token(api_client):
    os.environ["ADMIN_TOKEN"] = "let-me-in"
    try:
        r = api_client.post("/reload", headers={"X-Admin-Token": "let-me-in"})
        assert r.status_code == 200, r.text
        body = r.json()
        assert body["status"] == "ok"
        assert body["model_loaded"] is True
        assert body["model_source"] == "pickle"
    finally:
        os.environ.pop("ADMIN_TOKEN", None)
