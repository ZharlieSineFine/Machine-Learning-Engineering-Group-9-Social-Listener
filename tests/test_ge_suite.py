#Unit tests for the Great Expectations suite.

from __future__ import annotations

import pandas as pd
import pytest

from data.expectations.reviews_suite import (
    REQUIRED_COLUMNS,
    VALID_LABELS,
    validate_reviews,
)


def _good_df(n: int = 6) -> pd.DataFrame:
    rows = []
    labels = ["positive", "neutral", "negative"]
    for i in range(n):
        rows.append({
            "text": f"review number {i}",
            "label": labels[i % 3],
            "rating": (i % 5) + 1,
            "source": "google",
            "restaurant": "R",
            "location": "KL",
        })
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_good_batch_passes():
    result = validate_reviews(_good_df())
    assert result.success, f"expected pass, got failures: {result.failures}"
    assert result.n_rows == 6


def test_missing_column_fails():
    df = _good_df().drop(columns=["label"])
    result = validate_reviews(df)
    assert not result.success
    types = {f["expectation"] for f in result.failures}
    assert "expect_column_to_exist" in types


def test_null_text_fails():
    df = _good_df()
    df.loc[0, "text"] = None
    result = validate_reviews(df)
    assert not result.success
    types = {f["expectation"] for f in result.failures}
    assert "expect_column_values_to_not_be_null" in types


def test_unknown_label_fails():
    df = _good_df()
    df.loc[2, "label"] = "garbage"
    result = validate_reviews(df)
    assert not result.success
    types = {f["expectation"] for f in result.failures}
    assert "expect_column_values_to_be_in_set" in types


def test_rating_out_of_range_fails():
    df = _good_df()
    df.loc[1, "rating"] = 7  # outside 1..5
    result = validate_reviews(df)
    assert not result.success
    types = {f["expectation"] for f in result.failures}
    assert "expect_column_values_to_be_between" in types


def test_raise_for_status_raises_on_failure():
    df = _good_df()
    df.loc[0, "label"] = "bogus"
    result = validate_reviews(df)
    with pytest.raises(ValueError, match="reviews_suite validation failed"):
        result.raise_for_status()


def test_raise_for_status_silent_on_success():
    validate_reviews(_good_df()).raise_for_status()  # no-op


def test_text_below_min_length_fails():
    df = _good_df()
    df.loc[0, "text"] = "ok"  # below MIN_TEXT_LEN (=5)
    result = validate_reviews(df)
    assert not result.success
    types = {f["expectation"] for f in result.failures}
    assert "expect_column_value_lengths_to_be_between" in types


def test_text_above_max_length_fails():
    df = _good_df()
    df.loc[0, "text"] = "x" * 20_000  # above MAX_TEXT_LEN (=10_000)
    result = validate_reviews(df)
    assert not result.success
    types = {f["expectation"] for f in result.failures}
    assert "expect_column_value_lengths_to_be_between" in types


def test_pure_emoji_or_numeric_text_fails():
    """No letters at all → regex check trips."""
    df = _good_df()
    df.loc[0, "text"] = "12345 67890"  # no letters
    df.loc[1, "text"] = "!!!!!!!!!!"
    result = validate_reviews(df)
    assert not result.success
    types = {f["expectation"] for f in result.failures}
    assert "expect_column_values_to_match_regex" in types


def test_missing_label_class_fails():
    """If a batch has only 2 of 3 labels, cardinality check trips."""
    df = pd.DataFrame([
        {"text": "good food", "label": "positive", "rating": 5, "source": "g",
         "restaurant": "r", "location": "kl"},
        {"text": "bad food",  "label": "negative", "rating": 1, "source": "g",
         "restaurant": "r", "location": "kl"},
    ], columns=REQUIRED_COLUMNS)
    result = validate_reviews(df, check_language=False)  # short rows, skip lang
    assert not result.success
    types = {f["expectation"] for f in result.failures}
    assert "expect_column_distinct_values_to_contain_set" in types


def test_non_supported_language_fails():
    """A batch that's mostly French / Chinese trips the language gate."""
    rows = []
    for i in range(20):
        text = "Le restaurant est tres bon et le service excellent" if i % 2 else \
               "服务很好食物美味餐厅环境也不错值得推荐"
        rows.append({
            "text": text, "label": ["positive", "neutral", "negative"][i % 3],
            "rating": 4, "source": "g", "restaurant": "r", "location": "kl",
        })
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    result = validate_reviews(df)
    assert not result.success
    types = {f["expectation"] for f in result.failures}
    assert "expect_language_distribution" in types


def test_language_check_can_be_disabled():
    """check_language=False short-circuits language detection (useful for tests + offline runs)."""
    rows = [{
        "text": "Le restaurant est tres bon",
        "label": ["positive", "neutral", "negative"][i % 3],
        "rating": 4, "source": "g", "restaurant": "r", "location": "kl",
    } for i in range(6)]
    df = pd.DataFrame(rows, columns=REQUIRED_COLUMNS)
    result = validate_reviews(df, check_language=False)
    # With language check off the only thing remaining that could fail is... nothing.
    assert result.success, f"expected pass with check_language=False: {result.failures}"
