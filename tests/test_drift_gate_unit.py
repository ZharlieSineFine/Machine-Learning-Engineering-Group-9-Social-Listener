"""Unit tests for the Step 10 drift / recall_neg gate logic.

We exercise `evaluate()` with FAKE conn + minio (recorders) and a stub model.
No real DB / S3 needed.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List

import pandas as pd
import pytest

pytest.importorskip("evidently")  # drift checks need Evidently; skip cleanly if it's absent

from monitoring.drift_checks import (
    DEFAULT_RECALL_NEG_DROP_THRESHOLD,
    PromotionBlocked,
    compute_model_f1,
    compute_model_recall_neg,
    evaluate,
)


# --- fakes -----------------------------------------------------------------

class FakeMinIO:
    def __init__(self):
        self.uploads = []

    def put_object(self, Bucket, Key, Body, ContentType=None):
        self.uploads.append({"bucket": Bucket, "key": Key, "size": len(Body)})


class FakeCursor:
    def __init__(self, parent):
        self.parent = parent

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def execute(self, sql, params=None):
        self.parent.executed.append((sql, params))
        self._last = params

    def fetchone(self):
        return (len(self.parent.executed),)  # fake row id


class FakeConn:
    def __init__(self):
        self.executed: List = []

    def cursor(self):
        return FakeCursor(self)


@dataclass
class StubModel:
    """Predicts the same label for every input — handy for forcing F1 drops."""
    label: str

    def predict(self, texts):
        return [self.label] * len(texts)


# --- helpers ---------------------------------------------------------------

def _df(rows):
    return pd.DataFrame(rows, columns=["text", "label", "rating", "source"])


def _balanced(n: int, label_seed: str = None) -> pd.DataFrame:
    rows = []
    labels = ["positive", "neutral", "negative"]
    for i in range(n):
        rows.append({
            "text": f"this is review number {i}",
            "label": label_seed or labels[i % 3],
            "rating": (i % 5) + 1,
            "source": "google",
        })
    return _df(rows)


# --- tests -----------------------------------------------------------------

def test_compute_model_f1_perfect_predictor():
    df = _balanced(30)

    class Perfect:
        def predict(self, texts):
            # mirror the input back as the same label distribution
            return df["label"].tolist()[:len(texts)]

    assert compute_model_f1(Perfect(), df) == pytest.approx(1.0)


def test_compute_model_f1_always_wrong():
    df = _balanced(30)
    # Model always predicts "positive" — should be ~1/3 of the time correct.
    f1 = compute_model_f1(StubModel("positive"), df)
    assert 0.0 < f1 < 0.5


def test_compute_model_recall_neg_perfect_on_negatives():
    df = _balanced(30)

    class Perfect:
        def predict(self, texts):
            return df["label"].tolist()[:len(texts)]

    assert compute_model_recall_neg(Perfect(), df) == pytest.approx(1.0)


def test_compute_model_recall_neg_zero_when_negatives_missed():
    df = _balanced(30)
    assert compute_model_recall_neg(StubModel("positive"), df) == pytest.approx(0.0)


def test_evaluate_no_model_does_not_block_on_zero_drift():
    """Same df on both sides → drift = 0, no recall check, no block."""
    df = _balanced(60)
    result = evaluate(df, df, FakeConn(), FakeMinIO())
    assert result["blocked_promotion"] is False
    assert result["reference_recall_neg"] is None
    assert result["current_recall_neg"] is None


def test_evaluate_blocks_when_recall_neg_drops():
    """Model perfect on reference, misses negatives on current → block."""
    reference = _balanced(60)
    current = _balanced(60)

    class RecallDropModel:
        """Predicts perfectly on reference, 'positive' on current (recall_neg = 0)."""
        def __init__(self):
            self._mode = "ref"

        def predict(self, texts):
            if self._mode == "ref":
                self._mode = "cur"
                return reference["label"].tolist()[:len(texts)]
            return ["positive"] * len(texts)

    result = evaluate(reference, current, FakeConn(), FakeMinIO(), model=RecallDropModel())
    assert result["reference_recall_neg"] > result["current_recall_neg"]
    assert result["recall_neg_drop"] > DEFAULT_RECALL_NEG_DROP_THRESHOLD
    assert result["recall_neg_blocks"] is True
    assert result["blocked_promotion"] is True


def test_evaluate_raise_on_block_actually_raises():
    reference = _balanced(60)
    current = _balanced(60)

    class RecallDropModel:
        def __init__(self):
            self._mode = "ref"

        def predict(self, texts):
            if self._mode == "ref":
                self._mode = "cur"
                return reference["label"].tolist()[:len(texts)]
            return ["positive"] * len(texts)

    with pytest.raises(PromotionBlocked, match="recall_neg_drop"):
        evaluate(
            reference, current, FakeConn(), FakeMinIO(),
            model=RecallDropModel(), raise_on_block=True,
        )


def test_evaluate_does_not_block_when_model_stable():
    """Same model, same data on both sides → recall_neg identical → no block."""
    df = _balanced(60)

    class Perfect:
        def predict(self, texts):
            return df["label"].tolist()[:len(texts)]

    result = evaluate(df, df, FakeConn(), FakeMinIO(), model=Perfect())
    assert result["recall_neg_drop"] == pytest.approx(0.0)
    assert result["blocked_promotion"] is False


def test_evaluate_uploads_report_even_when_blocked():
    """Report must be uploaded BEFORE the raise so failures stay debuggable."""
    reference = _balanced(60)
    current = _balanced(60)
    minio = FakeMinIO()

    class RecallDropModel:
        def __init__(self):
            self._mode = "ref"

        def predict(self, texts):
            if self._mode == "ref":
                self._mode = "cur"
                return reference["label"].tolist()[:len(texts)]
            return ["positive"] * len(texts)

    with pytest.raises(PromotionBlocked):
        evaluate(reference, current, FakeConn(), minio,
                 model=RecallDropModel(), raise_on_block=True)

    assert len(minio.uploads) == 1, "the HTML must be uploaded even on block"
    assert minio.uploads[0]["bucket"] == "monitoring"
