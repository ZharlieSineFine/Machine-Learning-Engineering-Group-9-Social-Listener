"""Unit tests for models/splits.py.

Covers both split contracts:
    - ``split_gold`` — fixed ``review_date`` cutoffs for gold exports
    - ``train_val_test_oot_split`` — temporal OOT hold-out on Silver ``date``
"""
from __future__ import annotations

import pandas as pd
import pytest

from models.splits import (
    DEMO_CUTOFF,
    OOT_CUTOFF,
    DataSplit,
    GoldSplits,
    read_dataset,
    split_gold,
    train_val_test_oot_split,
)

LABELS = ("positive", "neutral", "negative")


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


def _frame(n: int, start: str = "2020-01-01", with_dates: bool = True) -> pd.DataFrame:
    """A frame of ``n`` rows with unique text, round-robin labels, ascending daily dates."""
    if with_dates:
        dates = [str(d) for d in pd.date_range(start, periods=n, freq="D")]
    else:
        dates = [None] * n
    return pd.DataFrame(
        {
            "text": [f"review number {i}" for i in range(n)],
            "label": [LABELS[i % len(LABELS)] for i in range(n)],
            "date": dates,
        }
    )


def test_oot_holds_out_most_recent_by_time():
    df = _frame(100)
    s = train_val_test_oot_split(df, oot_frac=0.2, val_frac=0.15, test_frac=0.15, seed=42)
    assert isinstance(s, DataSplit)
    assert len(s.oot) > 0
    oot_dates = pd.to_datetime(s.oot["date"])
    in_time = pd.concat([s.train, s.val, s.test])
    in_time_dates = pd.to_datetime(in_time["date"])
    assert oot_dates.min() > in_time_dates.max()
    assert oot_dates.min() == pd.Timestamp(str(s.cutoff_date))


def test_splits_partition_all_rows_without_loss_or_dup():
    df = _frame(60)
    s = train_val_test_oot_split(df, seed=0)
    everything = pd.concat([s.train, s.val, s.test, s.oot])
    assert len(everything) == len(df)
    assert set(everything["text"]) == set(df["text"])


def test_null_dates_never_land_in_oot():
    df = _frame(50)
    df.loc[::5, "date"] = None
    s = train_val_test_oot_split(df, oot_frac=0.3, seed=1)
    null_texts = set(df[df["date"].isna()]["text"])
    assert set(s.oot["text"]).isdisjoint(null_texts)


def test_rows_sharing_a_timestamp_are_not_split():
    days = pd.date_range("2020-01-01", periods=30, freq="D")
    df = pd.DataFrame(
        {
            "text": [f"r{i}" for i in range(60)],
            "label": [LABELS[i % 3] for i in range(60)],
            "date": [str(d) for d in days for _ in range(2)],
        }
    )
    s = train_val_test_oot_split(df, oot_frac=0.25, seed=2)
    oot_stamps = set(pd.to_datetime(s.oot["date"]))
    in_time_stamps = set(pd.to_datetime(pd.concat([s.train, s.val, s.test])["date"]))
    assert oot_stamps.isdisjoint(in_time_stamps)
    assert max(in_time_stamps) < min(oot_stamps)


def test_no_date_column_degrades_to_plain_split():
    df = _frame(60).drop(columns=["date"])
    s = train_val_test_oot_split(df)
    assert len(s.oot) == 0
    assert s.cutoff_date is None
    assert len(s.train) + len(s.val) + len(s.test) == 60


def test_all_null_dates_degrades_to_plain_split():
    df = _frame(40, with_dates=False)
    s = train_val_test_oot_split(df)
    assert len(s.oot) == 0
    assert len(s.train) + len(s.val) + len(s.test) == 40


def test_single_date_cannot_form_oot():
    df = pd.DataFrame(
        {
            "text": [f"r{i}" for i in range(20)],
            "label": [LABELS[i % 3] for i in range(20)],
            "date": ["2021-01-01"] * 20,
        }
    )
    s = train_val_test_oot_split(df, oot_frac=0.2)
    assert len(s.oot) == 0
    assert len(s.train) + len(s.val) + len(s.test) == 20


def test_deterministic_for_same_seed():
    df = _frame(80)
    a = train_val_test_oot_split(df, seed=7)
    b = train_val_test_oot_split(df, seed=7)
    assert a.train["text"].tolist() == b.train["text"].tolist()
    assert a.oot["text"].tolist() == b.oot["text"].tolist()


def test_stratification_keeps_all_classes():
    df = _frame(90)
    s = train_val_test_oot_split(df, oot_frac=0.0, val_frac=0.2, test_frac=0.2, seed=3)
    for part in (s.train, s.val, s.test):
        assert set(part["label"]) == set(LABELS)


def test_summary_reports_counts_and_ranges():
    df = _frame(50)
    s = train_val_test_oot_split(df, oot_frac=0.2, seed=4)
    summary = s.summary()
    assert summary["n_train"] + summary["n_val"] + summary["n_test"] + summary["n_oot"] == 50
    assert summary["cutoff_date"] is not None
    assert summary["oot_dates"] is not None


def test_invalid_fractions_raise():
    df = _frame(20)
    with pytest.raises(ValueError):
        train_val_test_oot_split(df, oot_frac=1.0)
    with pytest.raises(ValueError):
        train_val_test_oot_split(df, val_frac=0.6, test_frac=0.6)


def test_read_dataset_concatenates(tmp_path):
    a, b = _frame(5), _frame(7)
    pa, pb = tmp_path / "a.csv", tmp_path / "b.csv"
    a.to_csv(pa, index=False)
    b.to_csv(pb, index=False)
    out = read_dataset([str(pa), str(pb)])
    assert len(out) == 12
    assert list(out.columns) == ["text", "label", "date"]
