"""Shared sentiment metrics for training scripts and MLflow logging.

Metric names match ``notebooks/TUNING_NOTEBOOKS_INSTRUCTIONS.md`` and
``scripts/compare_mlflow_models.py`` (``{split}_f1_negative``, etc.).
"""
from __future__ import annotations

import json
import statistics
import time
from typing import Callable, Optional, Sequence

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, f1_score

LABELS = ["negative", "neutral", "positive"]
EVAL_SPLITS = ("val", "test", "oot")
INFERENCE_BATCH_SIZE = 32

# Logged to MLflow when present in the training metrics dict.
MLFLOW_SPLIT_METRICS = (
    "f1_negative",
    "recall_negative",
    "precision_negative",
    "f1_macro",
    "f1_neutral",
    "f1_positive",
)
MLFLOW_EXTRA_METRICS = ("inference_latency_ms",)


def classification_metrics(
    y_true,
    y_pred,
    *,
    labels: Sequence[str] = LABELS,
) -> dict:
    """Return per-split-style metrics (unprefixed) plus classification report."""
    report = classification_report(
        y_true, y_pred, labels=labels, output_dict=True, zero_division=0
    )
    neg = report["negative"]
    f1_neg = float(neg["f1-score"])
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_negative": f1_neg,
        "f1_neg": f1_neg,
        "recall_negative": float(neg["recall"]),
        "recall_neg": float(neg["recall"]),
        "precision_negative": float(neg["precision"]),
        "precision_neg": float(neg["precision"]),
        "f1_neutral": float(report["neutral"]["f1-score"]),
        "f1_positive": float(report["positive"]["f1-score"]),
        "report": report,
    }


def split_metrics(
    y_true,
    y_pred,
    split: str,
    *,
    labels: Sequence[str] = LABELS,
) -> dict:
    """Notebook-style metrics for one eval split (``test_f1_negative``, etc.)."""
    base = classification_metrics(y_true, y_pred, labels=labels)
    out = {
        f"{split}_f1_negative": base["f1_negative"],
        f"{split}_recall_negative": base["recall_negative"],
        f"{split}_precision_negative": base["precision_negative"],
        f"{split}_f1_macro": base["f1_macro"],
        f"{split}_f1_neutral": base["f1_neutral"],
        f"{split}_f1_positive": base["f1_positive"],
        f"{split}_accuracy": base["accuracy"],
    }
    return out


def train_metadata(train_df: pd.DataFrame) -> dict:
    """Training-set size and label counts for MLflow params."""
    counts = train_df["label"].value_counts().to_dict()
    return {
        "training_data_size": int(len(train_df)),
        "class_distribution": json.dumps({k: int(counts[k]) for k in sorted(counts)}),
    }


def measure_inference_latency_ms(
    predict_batch: Callable[[list[str]], object],
    texts: list[str],
    *,
    batch_size: int = INFERENCE_BATCH_SIZE,
    warmup: int = 1,
    repeats: int = 3,
) -> float:
    """Median wall-clock ms to score one batch of ``batch_size`` texts."""
    if not texts:
        return 0.0
    batch = list(texts[:batch_size])
    while len(batch) < batch_size:
        batch.extend(texts)
    batch = batch[:batch_size]

    for _ in range(warmup):
        predict_batch(batch)

    timings: list[float] = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        predict_batch(batch)
        timings.append((time.perf_counter() - t0) * 1000.0)
    return float(statistics.median(timings))


def mlflow_metrics(metrics: dict) -> dict:
    """Select split-prefixed metrics + latency for MLflow logging."""
    out: dict[str, float] = {}
    for split in EVAL_SPLITS:
        for name in MLFLOW_SPLIT_METRICS:
            key = f"{split}_{name}"
            val = metrics.get(key)
            if val is not None:
                out[key] = float(val)
    for key in MLFLOW_EXTRA_METRICS:
        val = metrics.get(key)
        if val is not None:
            out[key] = float(val)
    return out


def mlflow_training_params(metrics: dict) -> dict:
    """Standard training params logged by both entrypoints."""
    params = {
        "neg_threshold": metrics.get("neg_threshold", ""),
        "training_data_size": metrics.get("training_data_size", ""),
        "class_distribution": metrics.get("class_distribution", ""),
        "n_train": metrics.get("n_train", ""),
        "n_val": metrics.get("n_val", ""),
        "n_test": metrics.get("n_test", ""),
        "n_oot": metrics.get("n_oot", ""),
        "inference_batch_size": metrics.get("inference_batch_size", INFERENCE_BATCH_SIZE),
    }
    if metrics.get("cutoff_date") is not None:
        params["oot_cutoff_date"] = metrics["cutoff_date"]
    return params


def headline_from_test(metrics: dict) -> dict:
    """Map ``test_*`` keys to legacy unprefixed fields used by ``TrainResult``."""
    prefix = "test_"
    mapping = {
        "f1_macro": "f1_macro",
        "f1_negative": "f1_neg",
        "precision_negative": "precision_neg",
        "recall_negative": "recall_neg",
        "f1_neutral": "f1_neutral",
        "f1_positive": "f1_positive",
    }
    out: dict = {}
    for src, dst in mapping.items():
        key = f"{prefix}{src}"
        if key in metrics:
            out[dst] = metrics[key]
    if "test_f1_negative" in metrics:
        out["f1_negative"] = metrics["test_f1_negative"]
        out["f1_neg"] = metrics["test_f1_negative"]
    if "test_f1_macro" in metrics:
        out["f1_macro"] = metrics["test_f1_macro"]
    if "test_precision_negative" in metrics:
        out["precision_neg"] = metrics["test_precision_negative"]
    if "test_recall_negative" in metrics:
        out["recall_neg"] = metrics["test_recall_negative"]
    return out
