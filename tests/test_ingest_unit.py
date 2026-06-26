#Unit tests for the pure ingestion functions.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from data.ingest.ingest_reviews import (
    EXPECTED_COLUMNS,
    VALID_LABELS,
    load_and_validate,
    to_records,
)


def _write_csv(tmp: Path, rows: list[dict]) -> Path:
    p = tmp / "in.csv"
    pd.DataFrame(rows, columns=EXPECTED_COLUMNS).to_csv(p, index=False)
    return p


def test_load_validate_happy_path(tmp_path: Path):
    # All texts must clear MIN_TEXT_LEN (5) and contain at least one letter.
    p = _write_csv(tmp_path, [
        {"text": "great food",     "label": "positive", "rating": 5, "source": "g", "restaurant": "r", "location": "kl"},
        {"text": "average meal",   "label": "neutral",  "rating": 3, "source": "g", "restaurant": "r", "location": "kl"},
        {"text": "terrible value", "label": "negative", "rating": 1, "source": "g", "restaurant": "r", "location": "kl"},
    ])
    df = load_and_validate(p)
    assert list(df.columns) == EXPECTED_COLUMNS
    assert len(df) == 3
    assert set(df["label"]) <= VALID_LABELS


def test_drops_null_text(tmp_path: Path):
    p = _write_csv(tmp_path, [
        {"text": "decent food", "label": "positive", "rating": 5, "source": "g", "restaurant": "r", "location": "kl"},
        {"text": None,          "label": "positive", "rating": 5, "source": "g", "restaurant": "r", "location": "kl"},
        {"text": "",            "label": "positive", "rating": 5, "source": "g", "restaurant": "r", "location": "kl"},
    ])
    df = load_and_validate(p)
    assert len(df) == 1


def test_drops_short_or_emoji_text(tmp_path: Path):
    """The soft cleaner mirrors the GE length + regex checks."""
    p = _write_csv(tmp_path, [
        {"text": "good meal here",  "label": "positive", "rating": 5, "source": "g", "restaurant": "r", "location": "kl"},
        {"text": "ok",              "label": "neutral",  "rating": 3, "source": "g", "restaurant": "r", "location": "kl"},  # too short
        {"text": "👍👍👍👍👍",       "label": "positive", "rating": 5, "source": "g", "restaurant": "r", "location": "kl"},  # no letters
        {"text": "1234567890",      "label": "neutral",  "rating": 3, "source": "g", "restaurant": "r", "location": "kl"},  # no letters
    ])
    df = load_and_validate(p)
    assert len(df) == 1
    assert df.iloc[0]["text"] == "good meal here"


def test_drops_invalid_labels(tmp_path: Path):
    p = _write_csv(tmp_path, [
        {"text": "great food",  "label": "positive", "rating": 5, "source": "g", "restaurant": "r", "location": "kl"},
        {"text": "bad service", "label": "garbage",  "rating": 1, "source": "g", "restaurant": "r", "location": "kl"},
        {"text": "tasty stuff", "label": "POSITIVE", "rating": 5, "source": "g", "restaurant": "r", "location": "kl"},  # case-sensitive
    ])
    df = load_and_validate(p)
    assert len(df) == 1
    assert df.iloc[0]["text"] == "great food"


def test_raises_on_missing_columns(tmp_path: Path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"text": ["a"], "label": ["positive"]}).to_csv(p, index=False)
    with pytest.raises(ValueError, match="missing columns"):
        load_and_validate(p)


def test_to_records_shape(tmp_path: Path):
    p = _write_csv(tmp_path, [
        {"text": "lovely meal", "label": "positive", "rating": 5, "source": "g", "restaurant": "r", "location": "kl"},
    ])
    df = load_and_validate(p)
    records = to_records(df)
    assert len(records) == 1
    assert len(records[0]) == len(EXPECTED_COLUMNS)
    assert records[0][0] == "lovely meal"  # text
    assert records[0][1] == "positive"     # label
