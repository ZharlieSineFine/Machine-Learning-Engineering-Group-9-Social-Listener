"""Unit tests for models/metrics.py."""
from __future__ import annotations

from models.metrics import (
    classification_metrics,
    mlflow_metrics,
    split_metrics,
    train_metadata,
)
import pandas as pd


def test_classification_metrics_includes_aliases():
    y_true = ["negative", "positive", "neutral", "negative"]
    y_pred = ["negative", "positive", "neutral", "neutral"]
    metrics = classification_metrics(y_true, y_pred)
    assert metrics["f1_negative"] == metrics["f1_neg"]
    assert metrics["recall_negative"] == metrics["recall_neg"]
    assert 0.0 <= metrics["accuracy"] <= 1.0


def test_split_metrics_uses_notebook_prefixes():
    y_true = ["negative", "positive"]
    y_pred = ["negative", "positive"]
    metrics = split_metrics(y_true, y_pred, "test")
    assert metrics["test_f1_negative"] == 1.0
    assert metrics["test_f1_macro"] == 1.0
    assert metrics["test_f1_neutral"] == 0.0


def test_mlflow_metrics_extracts_prefixed_keys():
    raw = {
        "test_f1_negative": 0.6,
        "val_f1_macro": 0.7,
        "oot_f1_negative": 0.5,
        "test_f1_positive": 0.8,
        "test_f1_neutral": 0.4,
        "inference_latency_ms": 12.5,
        "f1_weighted": 0.75,
    }
    logged = mlflow_metrics(raw)
    assert logged["test_f1_negative"] == 0.6
    assert logged["val_f1_macro"] == 0.7
    assert logged["oot_f1_negative"] == 0.5
    assert logged["test_f1_positive"] == 0.8
    assert logged["inference_latency_ms"] == 12.5
    assert "f1_weighted" not in logged


def test_train_metadata_serializes_class_counts():
    df = pd.DataFrame({"label": ["negative", "neutral", "neutral", "positive"]})
    meta = train_metadata(df)
    assert meta["training_data_size"] == 4
    assert '"neutral": 2' in meta["class_distribution"]
