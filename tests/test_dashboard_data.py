"""Unit tests for dashboard/data.py — pure transformations.

The Streamlit `app.py` itself is smoke-tested in `test_dashboard_app.py` via
streamlit.testing.v1.AppTest. Here we focus on the transformation logic.
"""
from __future__ import annotations

import pandas as pd
import pytest

from dashboard.data import (
    fetch_drift_html,
    list_mlflow_runs,
    load_reviews,
    negative_word_counts,
    sentiment_timeline,
)



def test_load_reviews_csv_fallback_returns_sample():
    df = load_reviews(dsn=None)
    assert not df.empty
    assert {"text", "label"}.issubset(df.columns)


def test_load_reviews_csv_returns_empty_when_missing_file(tmp_path):
    df = load_reviews(dsn=None, csv_path=tmp_path / "nope.csv")
    assert df.empty


def test_load_reviews_falls_back_when_postgres_unreachable(tmp_path):
    """An obviously-bad DSN should not crash the app — falls back to CSV."""
    df = load_reviews(dsn="postgresql://nobody:nope@127.0.0.1:1/none")
    assert not df.empty  # falls back to the bundled sample


def _df_with_timestamps(n: int) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2025-01-01")
    for i in range(n):
        rows.append({
            "text": f"r {i}",
            "label": ["positive", "neutral", "negative"][i % 3],
            "ingested_at": base + pd.Timedelta(hours=i),
        })
    return pd.DataFrame(rows)


def test_timeline_groups_by_day():
    df = _df_with_timestamps(48)  # 2 days worth
    out = sentiment_timeline(df, freq="D")
    assert len(out) == 2
    assert set(out.columns) == {"period", "n_reviews", "pct_positive", "pct_negative", "pct_neutral"}
    assert out["n_reviews"].sum() == 48


def test_timeline_synthesises_timestamps_when_missing():
    """CSV sample has no `ingested_at` — function should still return rows."""
    df = pd.DataFrame({
        "text": ["a", "b", "c"],
        "label": ["positive", "neutral", "negative"],
    })
    out = sentiment_timeline(df, freq="D")
    assert not out.empty


def test_timeline_pct_positive_is_in_zero_hundred_range():
    df = _df_with_timestamps(30)
    out = sentiment_timeline(df, freq="D")
    assert ((out["pct_positive"] >= 0) & (out["pct_positive"] <= 100)).all()
    assert ((out["pct_negative"] >= 0) & (out["pct_negative"] <= 100)).all()


def test_negative_word_counts_strips_stopwords_and_short_tokens():
    df = pd.DataFrame({
        "text": [
            "the food was absolutely terrible service slow",   # negative
            "amazing food great service",                       # positive — ignored
            "soup cold burnt rice cold service slow",            # negative
        ],
        "label": ["negative", "positive", "negative"],
    })
    counts = negative_word_counts(df, top_n=10)
    assert "the" not in counts        # stopword
    assert "was" not in counts        # stopword
    assert "is" not in counts         # stopword
    assert counts["cold"] >= 2
    assert counts["service"] >= 2
    assert counts["slow"] >= 2


def test_negative_word_counts_returns_empty_on_no_negatives():
    df = pd.DataFrame({"text": ["a", "b"], "label": ["positive", "neutral"]})
    assert negative_word_counts(df) == {}


def test_negative_word_counts_returns_empty_on_missing_columns():
    df = pd.DataFrame({"foo": [1]})
    assert negative_word_counts(df) == {}


def test_list_mlflow_runs_returns_empty_when_uri_unset(monkeypatch):
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert list_mlflow_runs(tracking_uri=None).empty


def test_list_mlflow_runs_returns_empty_when_unreachable(monkeypatch):
    # Don't sit through MLflow's default retry budget — fail fast.
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "0")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "1")
    df = list_mlflow_runs(
        experiment_names=["nope"],
        tracking_uri="http://127.0.0.1:1",
    )
    assert df.empty  # connection refused → graceful empty


def test_fetch_drift_html_parses_s3_url():
    class FakeMinIO:
        def __init__(self):
            self.called_with = None

        def get_object(self, Bucket, Key):
            self.called_with = (Bucket, Key)
            return {"Body": type("B", (), {"read": lambda self: b"<html>ok</html>"})()}

    fake = FakeMinIO()
    body = fetch_drift_html("s3://monitoring/2025-01-01/data_drift.html", fake)
    assert fake.called_with == ("monitoring", "2025-01-01/data_drift.html")
    assert body == b"<html>ok</html>"


def test_fetch_drift_html_rejects_non_s3_url():
    class _NoOp:
        def get_object(self, **kw):
            raise AssertionError("should not be called")

    with pytest.raises(ValueError, match="s3://"):
        fetch_drift_html("https://elsewhere/r.html", _NoOp())
