"""Unit tests for monitoring/drift_checks.py — no DB, no S3.

Verifies the pure compute_drift() returns reasonable shapes on synthetic
data where we know whether drift exists or not.
"""
from __future__ import annotations

import pandas as pd
import pytest

pytest.importorskip("evidently")  # drift checks need Evidently; skip cleanly if it's absent

from monitoring.drift_checks import (
    DEFAULT_DRIFT_THRESHOLD,
    DriftResult,
    compute_drift,
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
