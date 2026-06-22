"""Unit tests for online shadow inference (no Postgres required)."""
from __future__ import annotations

import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from models import train as train_mod
from models.baseline_sklearn import LABELS, train
from models.inference import ModelSet, SentimentModel

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_CSV = ROOT / "data" / "sample" / "reviews_sample.csv"


@pytest.fixture(scope="module")
def model_set():
    df = pd.read_csv(SAMPLE_CSV)
    prod, _ = train(df)
    return ModelSet(
        production=SentimentModel(
            model=prod,
            model_name="sentiment-baseline",
            model_version="1",
            stage="Production",
            source="pickle",
        ),
        shadow=SentimentModel(
            model=prod,
            model_name="sentiment-distilbert",
            model_version="1",
            stage="Staging",
            source="pickle",
        ),
    )


def test_predict_with_shadow_returns_production_label(model_set):
    from app.shadow import predict_with_shadow

    labels = predict_with_shadow(model_set, ["amazing food"], review_ids=[None])
    assert len(labels) == 1
    assert labels[0] in LABELS


def test_predict_with_shadow_logs_both_lanes(model_set):
    from app.shadow import predict_with_shadow

    conn = MagicMock()
    cursor = MagicMock()
    conn.cursor.return_value.__enter__.return_value = cursor

    with patch("app.shadow._should_log_predictions", return_value=True), patch(
        "psycopg2.connect", return_value=conn
    ):
        predict_with_shadow(model_set, ["terrible service"], review_ids=[42])

    conn.commit.assert_called_once()
    written = cursor.executemany.call_args[0][1]
    assert len(written) == 2
    stages = {row[5] for row in written}
    assert stages == {"Production", "Staging"}


@pytest.fixture(scope="module")
def api_client(tmp_path_factory):
    artifact = tmp_path_factory.mktemp("artifacts") / "baseline.pkl"
    os.environ.pop("MLFLOW_TRACKING_URI", None)
    os.environ.pop("SHADOW_MODEL_NAME", None)
    train_mod.run(data_path=SAMPLE_CSV, out_path=artifact)
    os.environ["MODEL_PICKLE_PATH"] = str(artifact)

    for mod in ["app.main", "app.model_loader", "app.shadow"]:
        sys.modules.pop(mod, None)
    from app.main import app
    return TestClient(app)


def test_health_reports_shadow_disabled_without_env(api_client):
    body = api_client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["shadow_model_loaded"] is False
