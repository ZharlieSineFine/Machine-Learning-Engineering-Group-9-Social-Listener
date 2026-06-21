"""Unit tests for models/gold_loader.py — Gold feature+label join and fallback."""
from __future__ import annotations

import pandas as pd

from data.ingest.ingest_reviews import REVIEW_DATE_PARTITION, REVIEW_ID_FIELD
from models.gold_loader import (
    TRAINING_COLUMNS,
    load_gold_training_frame,
    materialize_training_csv,
)


def _write_gold(root):
    fdir = root / "feature_store" / "review_date=2026-06-01"
    ldir = root / "label_store" / "review_date=2026-06-01"
    fdir.mkdir(parents=True)
    ldir.mkdir(parents=True)
    pd.DataFrame(
        {
            REVIEW_ID_FIELD: ["a", "b"],
            REVIEW_DATE_PARTITION: ["2026-06-01", "2026-06-01"],
            "text": ["good coffee", "bad service"],
        }
    ).to_parquet(fdir / "part.parquet", index=False)
    pd.DataFrame(
        {
            REVIEW_ID_FIELD: ["a", "b"],
            REVIEW_DATE_PARTITION: ["2026-06-01", "2026-06-01"],
            "label": ["positive", "negative"],
        }
    ).to_parquet(ldir / "part.parquet", index=False)


def test_load_gold_joins_feature_and_label(tmp_path):
    _write_gold(tmp_path)
    df = load_gold_training_frame(gold_root=tmp_path, fallback_csv=None)
    assert list(df.columns) == TRAINING_COLUMNS
    assert len(df) == 2
    assert set(df["label"]) == {"positive", "negative"}


def test_load_gold_falls_back_to_sample_csv_when_empty(tmp_path):
    fallback = tmp_path / "sample.csv"
    pd.DataFrame({"text": ["x"], "label": ["positive"]}).to_csv(fallback, index=False)
    df = load_gold_training_frame(gold_root=tmp_path / "empty", fallback_csv=fallback)
    assert "text" in df.columns and "label" in df.columns
    assert len(df) == 1


def test_materialize_writes_training_csv(tmp_path):
    _write_gold(tmp_path)
    out = materialize_training_csv(
        tmp_path / "train.csv", gold_root=tmp_path, fallback_csv=None
    )
    assert out.exists()
    assert len(pd.read_csv(out)) == 2
