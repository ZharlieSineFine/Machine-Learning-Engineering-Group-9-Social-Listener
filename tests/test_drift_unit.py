"""Unit tests for monitoring/drift_checks.py — no DB, no S3.

Verifies the pure compute_drift() returns reasonable shapes on synthetic
data where we know whether drift exists or not.
"""
from __future__ import annotations

import pandas as pd
import pytest

from monitoring.drift_checks import (
    DEFAULT_DRIFT_THRESHOLD,
    DriftResult,
    compute_drift,
    compute_drift_report,
)


def _make_df(n: int, label_dist: dict, source_dist: dict, rating_mean: float) -> pd.DataFrame:
    """Build a synthetic reviews-like dataframe with the requested distributions."""
    import numpy as np

    rng = np.random.default_rng(42)
    labels = rng.choice(list(label_dist), p=list(label_dist.values()), size=n)
    sources = rng.choice(list(source_dist), p=list(source_dist.values()), size=n)
    ratings = np.clip(rng.normal(rating_mean, 0.8, n), 1, 5)
    return pd.DataFrame({
        "text": [f"row {i}" for i in range(n)],
        "label": labels,
        "rating": ratings,
        "source": sources,
    })


def test_compute_drift_returns_drift_result():
    ref = _make_df(200, {"positive": 0.5, "neutral": 0.3, "negative": 0.2},
                   {"google": 0.6, "tripadvisor": 0.4}, rating_mean=4.0)
    cur = _make_df(150, {"positive": 0.5, "neutral": 0.3, "negative": 0.2},
                   {"google": 0.6, "tripadvisor": 0.4}, rating_mean=4.0)

    result = compute_drift(ref, cur)
    assert isinstance(result, DriftResult)
    assert result.n_reference == 200
    assert result.n_current == 150
    assert 0.0 <= result.drift_score <= 1.0
    assert isinstance(result.drifted_columns, list)
    assert len(result.html) > 1000  # the HTML report is non-trivial


def test_compute_drift_html_starts_with_doctype():
    ref = _make_df(50, {"positive": 0.5, "negative": 0.5}, {"google": 1.0}, 4.0)
    cur = _make_df(50, {"positive": 0.5, "negative": 0.5}, {"google": 1.0}, 4.0)
    result = compute_drift(ref, cur)
    # Evidently HTML may have leading whitespace before the doctype.
    stripped = result.html.lstrip()
    assert stripped[:30].lower().startswith((b"<!doctype html", b"<html"))


def test_compute_drift_detects_shifted_distribution():
    """Shifting all ratings up by 2 and flipping label dominance should drift."""
    ref = _make_df(300, {"positive": 0.7, "negative": 0.3},
                   {"google": 0.8, "tripadvisor": 0.2}, rating_mean=2.0)
    cur = _make_df(300, {"positive": 0.3, "negative": 0.7},
                   {"google": 0.2, "tripadvisor": 0.8}, rating_mean=4.5)

    result = compute_drift(ref, cur)
    # We expect at least one of (rating, source, label) to be flagged.
    assert result.drift_score > 0.0, "expected drift, got 0"
    assert result.drifted_columns, "expected at least one drifted column"


def test_is_blocking_threshold():
    r = DriftResult(html=b"", drift_score=0.6, drifted_columns=[], n_reference=1, n_current=1)
    assert r.is_blocking(threshold=0.5) is True
    assert r.is_blocking(threshold=0.7) is False


# ---------------------------------------------------------------------------
# PSI — compute_drift now uses the PSI stattest and surfaces per-column values
# ---------------------------------------------------------------------------
def test_compute_drift_populates_psi_by_column():
    ref = _make_df(300, {"positive": 0.5, "neutral": 0.3, "negative": 0.2},
                   {"google": 0.6, "tripadvisor": 0.4}, rating_mean=4.0)
    cur = _make_df(300, {"positive": 0.5, "neutral": 0.3, "negative": 0.2},
                   {"google": 0.6, "tripadvisor": 0.4}, rating_mean=4.0)

    result = compute_drift(ref, cur)
    # PSI is reported per monitored column (text_len excluded here; rating/source/label present).
    assert isinstance(result.psi_by_column, dict)
    assert result.psi_by_column, "expected per-column PSI values"
    assert all(isinstance(v, float) and v >= 0.0 for v in result.psi_by_column.values())


def test_compute_drift_psi_flags_shifted_columns():
    """A large distribution shift should push at least one column's PSI past 0.2."""
    ref = _make_df(400, {"positive": 0.8, "negative": 0.2},
                   {"google": 0.9, "tripadvisor": 0.1}, rating_mean=4.5)
    cur = _make_df(400, {"positive": 0.2, "negative": 0.8},
                   {"google": 0.1, "tripadvisor": 0.9}, rating_mean=1.8)

    result = compute_drift(ref, cur)
    assert result.drift_score > 0.0
    assert max(result.psi_by_column.values()) >= 0.2


# ---------------------------------------------------------------------------
# Prediction-distribution drift — compute_drift_report
# ---------------------------------------------------------------------------
class _FakeModel:
    """Predicts a fixed label distribution regardless of input text length."""

    def __init__(self, label: str):
        self._label = label

    def predict(self, texts):
        return [self._label] * len(texts)


def test_prediction_drift_detected_when_output_mix_shifts():
    ref = _make_df(200, {"positive": 0.5, "negative": 0.5}, {"google": 1.0}, 4.0)
    cur = _make_df(200, {"positive": 0.5, "negative": 0.5}, {"google": 1.0}, 4.0)
    # Reference predictions skew positive, current predictions skew negative.
    ref["prediction"] = ["positive"] * 180 + ["negative"] * 20
    cur["prediction"] = ["positive"] * 20 + ["negative"] * 180

    out = compute_drift_report(ref, cur, model=None)
    assert out["used_model"] is True
    assert out["prediction_drift_score"] is not None
    assert out["prediction_drift"] is True


def test_prediction_drift_uses_existing_current_prediction_column():
    """Table path: current already carries `prediction`; only reference is scored."""
    ref = _make_df(150, {"positive": 0.5, "negative": 0.5}, {"google": 1.0}, 4.0)
    cur = _make_df(150, {"positive": 0.5, "negative": 0.5}, {"google": 1.0}, 4.0)
    cur["prediction"] = ["negative"] * 150  # logged predictions, all negative

    out = compute_drift_report(ref, cur, model=_FakeModel("positive"))
    # Reference scored positive, current logged negative -> prediction drift.
    assert out["used_model"] is True
    assert out["prediction_drift"] is True
