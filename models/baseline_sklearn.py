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
from typing import Tuple

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

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


def train(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> Tuple[Pipeline, dict]:
    """Fit the baseline on `df` and return (fitted_pipeline, metrics_dict).

    `df` must have columns `text` and `label` (label in LABELS).
    """
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("df must have 'text' and 'label' columns")

    # TODO (member): improve the SPLIT STRATEGY.
    #   Right now we do a single random split. Better options:
    #     - StratifiedKFold for a more honest estimate on 1k rows.
    #     - GroupKFold on `restaurant` to test generalisation to unseen places.
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"].astype(str),
        df["label"],
        test_size=test_size,
        random_state=seed,
        stratify=df["label"],
    )

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    # TODO (member): expand the METRICS.
    #   The MLflow registry promotion gate (Phase 2) will need at least
    #   per-class precision/recall and a confusion matrix. Log them here
    #   so models/train.py can forward them to MLflow without re-computing.
    metrics = {
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "report": classification_report(y_test, y_pred, output_dict=True, zero_division=0),
    }
    return pipe, metrics
