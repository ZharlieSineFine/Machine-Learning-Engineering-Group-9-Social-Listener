#Baseline sentiment classifier: TF-IDF + LogisticRegression (the Phase 1 thin-slice model).
from __future__ import annotations

import re
import string
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import Pipeline

from models.splits import train_val_test_oot_split

LABELS = ["negative", "neutral", "positive"]
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
DEFAULT_NEG_THRESHOLD = 0.46


@dataclass
class TunedSentimentPipeline:
    #Serving wrapper around a fitted TF-IDF + LogReg pipeline; applies the tuned negative threshold.
    #Pickled into MLflow, so it must stay importable as models.baseline_sklearn.TunedSentimentPipeline.
    pipeline: Any
    neg_threshold: float = DEFAULT_NEG_THRESHOLD

    @staticmethod
    def _to_label(c: Any) -> str:
        if isinstance(c, str):
            return c
        try:
            return ID2LABEL[int(c)]
        except (KeyError, ValueError, TypeError):
            return str(c)

    def _neg_index(self, classes: list) -> int:
        for i, c in enumerate(classes):
            if self._to_label(c) == "negative":
                return i
        return 0

    def predict(self, X) -> np.ndarray:
        proba = self.pipeline.predict_proba(X)
        classes = list(self.pipeline.classes_)
        neg_i = self._neg_index(classes)
        out = []
        for row in proba:
            if row[neg_i] >= self.neg_threshold:
                out.append("negative")
            else:
                best = max(range(len(row)), key=lambda i: row[i] if i != neg_i else float("-inf"))
                out.append(self._to_label(classes[best]))
        return np.array(out)

    def predict_proba(self, X) -> np.ndarray:
        return self.pipeline.predict_proba(X)

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
    #Normalize noisy review text before tokenization.
    text = text.lower()
    text = _URL_RE.sub(" ", text)
    text = _MENTION_RE.sub(" ", text)
    for emoji, replacement in _EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji, replacement)
    text = text.translate(_PUNCT_TABLE)
    return _SPACE_RE.sub(" ", text).strip()


def build_pipeline() -> Pipeline:
    #Return an untrained TF-IDF + LogisticRegression pipeline (minimal: trains in <5s on 1k rows).
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
    #Score a fitted pipeline on a non-empty eval frame; None if the frame is empty.
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
    #Fit the baseline with a train/validation/test/OOT split; return (pipeline, metrics).
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
