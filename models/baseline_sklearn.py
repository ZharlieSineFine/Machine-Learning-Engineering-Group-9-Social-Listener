<<<<<<< HEAD
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
=======
"""Baseline sentiment classifier — TF-IDF + LogisticRegression.

This is the **thin-slice** model from WORKFLOW.md Phase 1. It exists to keep
the end-to-end pipeline (ingest -> train -> registry -> API -> dashboard)
green so nothing is blocked while the modelling team iterates.

Owner: Van (Modeler), Amelia (second pair).

The training entry point is `models/train.py` — this module only exposes:
    - build_pipeline()  : returns an untrained sklearn Pipeline
    - train(df)         : fits the pipeline, returns (pipeline, metrics)
    - LABELS            : the canonical label list

Members: search for `# TODO (member)` to find the spots where you should
improve the model. The smoke test does NOT depend on these improvements —
it only checks that the pipeline can fit + predict end to end.
"""
from __future__ import annotations

import re
import string
from typing import Optional, Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from models.splits import train_val_test_oot_split

LABELS = ["negative", "neutral", "positive"]

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"@\w+")
_SPACE_RE = re.compile(r"\s+")
_PUNCT_TABLE = str.maketrans({ch: " " for ch in string.punctuation})
_EMOJI_REPLACEMENTS = {
    "😀": " positive ",
    "😃": " positive ",
    "😄": " positive ",
    "😁": " positive ",
    "😊": " positive ",
    "😍": " positive ",
    "😋": " positive ",
    "👍": " positive ",
    "❤️": " positive ",
    "❤": " positive ",
    "😐": " neutral ",
    "😕": " negative ",
    "🙁": " negative ",
    "☹": " negative ",
    "😞": " negative ",
    "😡": " negative ",
    "👎": " negative ",
}


def _preprocess_text(text: str) -> str:
    """Normalize noisy review text before tokenization."""
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    for emoji, replacement in _EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji, replacement)
    text = text.translate(_PUNCT_TABLE)
    return _SPACE_RE.sub(" ", text).strip()


def build_pipeline() -> Pipeline:
    """Return an untrained TF-IDF + LogisticRegression pipeline.

    Intentionally minimal so the baseline trains in <5 seconds on 1k rows.
    """
    vectorizer = TfidfVectorizer(
        # TODO (member): tune the VECTORIZER.
        #   Try: ngram_range=(1, 2), min_df=2, max_df=0.95,
        #        sublinear_tf=True, max_features=20_000.
        #   For multilingual data: char n-grams (analyzer='char_wb',
        #   ngram_range=(3, 5)) often beat word n-grams on Malay/English mix.
        max_features=5_000,
        preprocessor=_preprocess_text,
        stop_words="english",
    )

    # TODO (member): swap the MODEL.
    #   Baselines worth trying before moving to DistilBERT:
    #     - LinearSVC (often beats LogReg on TF-IDF text)
    #     - ComplementNB (fast, robust on imbalanced classes)
    #     - SGDClassifier with class_weight='balanced'
    #   Then graduate to models/distilbert_finetune.py (Phase 2).
    classifier = LogisticRegression(
        max_iter=1_000,
        class_weight="balanced",
        n_jobs=-1,
    )

    return Pipeline([("tfidf", vectorizer), ("clf", classifier)])


def _evaluate(pipe: Pipeline, frame: pd.DataFrame) -> Optional[dict]:
    """Score a fitted pipeline on a non-empty eval frame; None if the frame is empty."""
    if frame.empty:
        return None
    y_true = frame["label"]
    y_pred = pipe.predict(frame["text"].astype(str))
    report = classification_report(
        y_true, y_pred, labels=LABELS, output_dict=True, zero_division=0
    )
    neg = report["negative"]
    return {
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_neg": float(neg["f1-score"]),
        "precision_neg": float(neg["precision"]),
        "recall_neg": float(neg["recall"]),
        "report": report,
    }


def train(
    df: pd.DataFrame,
    test_size: float = 0.2,
    seed: int = 42,
    *,
    oot_frac: float = 0.2,
    val_frac: float = 0.15,
) -> Tuple[Pipeline, dict]:
    """Fit the baseline with a train/validation/test/OOT split; return (pipeline, metrics).

    `df` must have `text` and `label` (label in LABELS). When `df` carries a `date` column,
    the most recent `oot_frac` of dated rows is held out as an out-of-time (OOT) set and
    scored separately (see models/splits.py); on date-less data OOT is empty and this is a
    plain stratified train/val/test split. `test_size` is the in-time *test* fraction (kept
    as the legacy parameter name).

    The model is fit on the train split only — `val` is reserved for tuning / model
    selection (the baseline doesn't tune yet). `f1_macro` / `f1_weighted` are the in-time
    *test* scores (backward-compatible keys); the OOT scores are reported under `*_oot`, and
    the test-vs-OOT gap is the headline signal of temporal drift.
    """
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("df must have 'text' and 'label' columns")

    split = train_val_test_oot_split(
        df, oot_frac=oot_frac, val_frac=val_frac, test_frac=test_size, seed=seed
    )

    pipe = build_pipeline()
    pipe.fit(split.train["text"].astype(str), split.train["label"])

    val_metrics = _evaluate(pipe, split.val)
    test_metrics = _evaluate(pipe, split.test)
    oot_metrics = _evaluate(pipe, split.oot)
    headline = test_metrics or val_metrics or {
        "f1_macro": 0.0,
        "f1_weighted": 0.0,
        "accuracy": 0.0,
        "f1_neg": 0.0,
        "precision_neg": 0.0,
        "recall_neg": 0.0,
        "report": {},
    }

    metrics = {
        "f1_macro": headline["f1_macro"],
        "f1_weighted": headline["f1_weighted"],
        "accuracy": headline["accuracy"],
        "f1_neg": headline["f1_neg"],
        "precision_neg": headline["precision_neg"],
        "recall_neg": headline["recall_neg"],
        "report": headline.get("report", {}),
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_test": int(len(split.test)),
        "n_oot": int(len(split.oot)),
        "cutoff_date": None if split.cutoff_date is None else str(split.cutoff_date),
    }
    if val_metrics:
        metrics["f1_macro_val"] = val_metrics["f1_macro"]
        metrics["f1_weighted_val"] = val_metrics["f1_weighted"]
    if oot_metrics:
        metrics["f1_macro_oot"] = oot_metrics["f1_macro"]
        metrics["f1_weighted_oot"] = oot_metrics["f1_weighted"]
        metrics["report_oot"] = oot_metrics["report"]
    return pipe, metrics
>>>>>>> origin/feature/full_flow
