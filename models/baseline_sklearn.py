"""Baseline sentiment classifier — TF-IDF + LogisticRegression.

Production defaults come from the winning ``logreg-final`` run in
``notebooks/01_tfidf_logreg_tuning.ipynb`` (GridSearchCV + val threshold tuning).

Owner: Van (Modeler), Amelia (second pair).

The training entry point is ``models/train.py`` — this module exposes:
    - build_pipeline()           : untrained sklearn Pipeline
    - TunedSentimentPipeline     : fitted pipeline + negative threshold
    - train(df)                  : fit, wrap, return (model, metrics)
    - LABELS                     : canonical label list
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from models.metrics import (
    INFERENCE_BATCH_SIZE,
    classification_metrics,
    measure_inference_latency_ms,
    split_metrics,
    train_metadata,
)
from models.splits import train_val_test_oot_split

LABELS = ["negative", "neutral", "positive"]
NEG_IDX = 0

# Tuned in notebook 01 (`logreg-final` / GridSearchCV on f1_negative).
TUNED_NGRAM_RANGE = (1, 3)
TUNED_MAX_FEATURES = 100_000
TUNED_C = 10.0
TUNED_CLASS_WEIGHT: str | None = None
DEFAULT_NEG_THRESHOLD = 0.46


def clean_text(text: str) -> str:
    """Match notebook 01 preprocessing before TF-IDF."""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def build_pipeline(
    ngram_range: tuple[int, int] = TUNED_NGRAM_RANGE,
    max_features: int = TUNED_MAX_FEATURES,
    C: float = TUNED_C,
    class_weight: str | None = TUNED_CLASS_WEIGHT,
    seed: int = 42,
) -> Pipeline:
    """Return an untrained TF-IDF + LogisticRegression pipeline."""
    vectorizer = TfidfVectorizer(
        ngram_range=ngram_range,
        max_features=max_features,
    )
    classifier = LogisticRegression(
        C=C,
        class_weight=class_weight,
        max_iter=1_000,
        random_state=seed,
        n_jobs=-1,
    )
    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


@dataclass
class TunedSentimentPipeline:
    """Fitted sklearn pipeline with the tuned negative-class threshold."""

    pipeline: Pipeline
    neg_threshold: float = DEFAULT_NEG_THRESHOLD
    neg_idx: int = NEG_IDX

    def _neg_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, self.neg_idx]

    def predict(self, X) -> np.ndarray:
        probs_neg = self._neg_proba(X)
        base = self.pipeline.predict(X)
        return np.where(probs_neg >= self.neg_threshold, LABELS[self.neg_idx], base)

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict_with_threshold(
        self, X, threshold: float | None = None
    ) -> np.ndarray:
        t = self.neg_threshold if threshold is None else threshold
        probs_neg = self._neg_proba(X)
        base = self.pipeline.predict(X)
        return np.where(probs_neg >= t, LABELS[self.neg_idx], base)


def _prepare_texts(texts: Union[pd.Series, list, np.ndarray]) -> list[str]:
    return [clean_text(t) for t in texts]


def _evaluate_split(
    model: TunedSentimentPipeline, frame: pd.DataFrame, split: str
) -> Optional[dict]:
    """Score a fitted model on a non-empty eval frame; None if the frame is empty."""
    if frame.empty:
        return None
    y_true = frame["label"]
    X = _prepare_texts(frame["text"].astype(str))
    y_pred = model.predict(X)
    return split_metrics(y_true, y_pred, split, labels=LABELS)


def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    neg_threshold: float = DEFAULT_NEG_THRESHOLD,
    *,
    oot_frac: float = 0.2,
    val_frac: float = 0.15,
) -> Tuple[TunedSentimentPipeline, dict]:
    """Fit the tuned baseline with a train/validation/test/OOT split; return (model, metrics).

    ``df`` must have ``text`` and ``label`` (label in LABELS). When ``df`` carries a
    ``date`` column, the most recent ``oot_frac`` of dated rows is held out as OOT and
    scored separately (see ``models/splits.py``); on date-less data OOT is empty.

    MLflow-facing metrics use notebook prefixes: ``test_f1_negative``, ``val_f1_macro``, etc.
    """
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("df must have 'text' and 'label' columns")

    split = train_val_test_oot_split(
        df, oot_frac=oot_frac, val_frac=val_frac, test_frac=test_size, seed=seed
    )

    X_train = _prepare_texts(split.train["text"].astype(str))
    y_train = split.train["label"]

    fitted = build_pipeline(seed=seed).fit(X_train, y_train)
    model = TunedSentimentPipeline(pipeline=fitted, neg_threshold=neg_threshold)

    metrics: dict = {
        "neg_threshold": float(neg_threshold),
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_test": int(len(split.test)),
        "n_oot": int(len(split.oot)),
        "cutoff_date": None if split.cutoff_date is None else str(split.cutoff_date),
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        "tuned_params": {
            "tfidf__ngram_range": TUNED_NGRAM_RANGE,
            "tfidf__max_features": TUNED_MAX_FEATURES,
            "clf__C": TUNED_C,
            "clf__class_weight": TUNED_CLASS_WEIGHT,
        },
        **train_metadata(split.train),
    }

    for split_name, frame in (("val", split.val), ("test", split.test), ("oot", split.oot)):
        split_scores = _evaluate_split(model, frame, split_name)
        if split_scores:
            metrics.update(split_scores)

    if not split.test.empty:
        texts = _prepare_texts(split.test["text"].astype(str))
        metrics["inference_latency_ms"] = measure_inference_latency_ms(
            lambda batch: model.predict(_prepare_texts(batch)),
            texts,
            batch_size=INFERENCE_BATCH_SIZE,
        )

    # Legacy unprefixed keys for TrainResult / smoke tests.
    if "test_f1_macro" in metrics:
        y_true = split.test["label"]
        y_pred = model.predict(_prepare_texts(split.test["text"].astype(str)))
        legacy = classification_metrics(y_true, y_pred, labels=LABELS)
        metrics["f1_macro"] = metrics["test_f1_macro"]
        metrics["f1_weighted"] = legacy["f1_weighted"]
        metrics["f1_negative"] = metrics["test_f1_negative"]
        metrics["f1_neg"] = metrics["test_f1_negative"]
        metrics["precision_neg"] = metrics["test_precision_negative"]
        metrics["recall_neg"] = metrics["test_recall_negative"]
        metrics["accuracy"] = metrics.get("test_accuracy", legacy["accuracy"])
    else:
        metrics.update({
            "f1_macro": 0.0,
            "f1_weighted": 0.0,
            "accuracy": 0.0,
            "f1_negative": 0.0,
            "f1_neg": 0.0,
            "precision_neg": 0.0,
            "recall_neg": 0.0,
        })

    if "oot_f1_macro" in metrics:
        metrics["f1_macro_oot"] = metrics["oot_f1_macro"]
    if "val_f1_macro" in metrics:
        metrics["f1_macro_val"] = metrics["val_f1_macro"]

    return model, metrics
