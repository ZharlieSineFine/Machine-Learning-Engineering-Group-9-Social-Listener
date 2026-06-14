"""Unit tests for models/splits.py."""
from __future__ import annotations

import pandas as pd
import pytest

from models.splits import DEMO_CUTOFF, OOT_CUTOFF, GoldSplits, split_gold


def _synthetic_gold(n: int = 200) -> pd.DataFrame:
    dates = pd.date_range("2021-06-01", periods=n, freq="D")
    labels = ["negative", "neutral", "positive"]
    return pd.DataFrame({
        "text": [f"review {i}" for i in range(n)],
        "label": [labels[i % 3] for i in range(n)],
        "review_date": dates,
    })


def test_split_gold_returns_all_partitions():
    splits = split_gold(_synthetic_gold())
    assert isinstance(splits, GoldSplits)
    for name, part in splits.as_dict().items():
        assert isinstance(part, pd.DataFrame)
        assert {"text", "label", "review_date"}.issubset(part.columns)


def test_split_gold_temporal_holdouts():
    df = _synthetic_gold(120)
    splits = split_gold(df, oot_cutoff=OOT_CUTOFF, demo_cutoff=DEMO_CUTOFF)

    assert (splits.demo["review_date"] >= pd.Timestamp(DEMO_CUTOFF)).all()
    assert (splits.oot["review_date"] >= pd.Timestamp(OOT_CUTOFF)).all()
    assert (splits.oot["review_date"] < pd.Timestamp(DEMO_CUTOFF)).all()
    assert (splits.train["review_date"] < pd.Timestamp(OOT_CUTOFF)).all()


def test_split_gold_rejects_missing_columns():
    with pytest.raises(ValueError, match="review_date"):
        split_gold(pd.DataFrame({"text": ["a"], "label": ["positive"]}))


def test_split_gold_no_rows_before_oot_raises():
    df = pd.DataFrame({
        "text": ["late review"],
        "label": ["positive"],
        "review_date": [DEMO_CUTOFF],
    })
    with pytest.raises(ValueError, match="No rows before OOT cutoff"):
        split_gold(df)
