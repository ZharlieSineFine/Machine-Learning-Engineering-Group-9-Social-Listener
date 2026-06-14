"""Baseline sentiment classifier — TF-IDF + LogisticRegression.

Production defaults come from the winning `logreg-final` run in
`notebooks/01_tfidf_logreg_tuning.ipynb` (GridSearchCV + val threshold tuning).

Owner: Van (Modeler), Amelia (second pair).

The training entry point is `models/train.py` — this module exposes:
    - build_pipeline()           : untrained sklearn Pipeline
    - TunedSentimentPipeline     : fitted pipeline + negative threshold
    - train(df)                  : fit, wrap, return (model, metrics)
    - LABELS                     : canonical label list
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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


def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    neg_threshold: float = DEFAULT_NEG_THRESHOLD,
) -> Tuple[TunedSentimentPipeline, dict]:
    """Fit the tuned baseline on `df` and return (model, metrics_dict).

    `df` must have columns `text` and `label` (label in LABELS).
    Metrics are computed with the tuned negative threshold applied.
    """
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("df must have 'text' and 'label' columns")

    X = _prepare_texts(df["text"].astype(str))
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    fitted = build_pipeline(seed=seed).fit(X_train, y_train)
    model = TunedSentimentPipeline(pipeline=fitted, neg_threshold=neg_threshold)
    y_pred = model.predict(X_test)

    report = classification_report(
        y_test, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    neg = report["negative"]
    metrics = {
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_neg": float(neg["f1-score"]),
        "precision_neg": float(neg["precision"]),
        "recall_neg": float(neg["recall"]),
        "neg_threshold": float(neg_threshold),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "report": report,
        "tuned_params": {
            "tfidf__ngram_range": TUNED_NGRAM_RANGE,
            "tfidf__max_features": TUNED_MAX_FEATURES,
            "clf__C": TUNED_C,
            "clf__class_weight": TUNED_CLASS_WEIGHT,
        },
    }
    return model, metrics
