"""Unit tests for models/distilbert_finetune.py.

The end-to-end "does it actually train" test is `tests/test_distilbert_slow.py`
and is gated by `RUN_SLOW=1` (or `pytest -m slow`).
"""
from __future__ import annotations

import sys
import types

import numpy as np
import pandas as pd
import pytest

from models.distilbert_finetune import (
    DEFAULT_MAX_LEN,
    ID2LABEL,
    LABEL2ID,
    LABELS,
    TrainConfig,
    _compute_metrics,
    _encode_dataset,
    train_distilbert,
)


def test_label_map_roundtrip():
    assert set(LABEL2ID) == set(LABELS)
    for lbl, idx in LABEL2ID.items():
        assert ID2LABEL[idx] == lbl


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.num_epochs >= 1
    assert cfg.max_length == DEFAULT_MAX_LEN
    assert cfg.max_steps == -1  # i.e. honor num_epochs unless overridden


def test_compute_metrics_shape():
    # 3-class logits, 4 examples
    logits = np.array(
        [[2.0, 0.1, 0.1],
         [0.1, 2.0, 0.1],
         [0.1, 0.1, 2.0],
         [0.5, 0.4, 0.1]],
    )
    labels = np.array([0, 1, 2, 0])
    metrics = _compute_metrics((logits, labels))
    assert {"accuracy", "f1_macro"}.issubset(metrics)
    assert metrics["accuracy"] == 1.0  # all argmaxes match
    assert 0.0 <= metrics["f1_macro"] <= 1.0


class DummyTokenizer:
    def __call__(self, texts, truncation, padding, max_length):
        batch_size = len(texts)
        return {
            "input_ids": [[1] * max_length for _ in range(batch_size)],
            "attention_mask": [[1] * max_length for _ in range(batch_size)],
        }


def _install_fake_datasets(monkeypatch):
    fake = types.ModuleType("datasets")

    class DummyDataset:
        def __init__(self, df):
            self._df = df
            self._data = []
            self.column_names = []

        @classmethod
        def from_pandas(cls, df, preserve_index=False):
            if not preserve_index:
                df = df.reset_index(drop=True)
            return cls(df)

        def map(self, fn, batched, remove_columns):
            mapped = fn({
                "text": self._df["text"].tolist(),
                "label_id": self._df["label_id"].tolist(),
            })
            self._data = []
            for i in range(len(mapped["labels"])):
                self._data.append({
                    "input_ids": np.array(mapped["input_ids"][i]),
                    "attention_mask": np.array(mapped["attention_mask"][i]),
                    "labels": np.array(mapped["labels"][i]),
                })
            self.column_names = list(mapped.keys())
            return self

        def set_format(self, type, columns):
            self.column_names = list(columns)
            return self

        def __len__(self):
            return len(self._data)

        def __getitem__(self, idx):
            return self._data[idx]

    fake.Dataset = DummyDataset
    monkeypatch.setitem(sys.modules, "datasets", fake)


def test_encode_dataset_returns_torch_tensors(monkeypatch):
    """Use a fake tokenizer so the unit test stays fast and dependency-free."""
    _install_fake_datasets(monkeypatch)
    tokenizer = DummyTokenizer()
    df = pd.DataFrame({
        "text": ["the food was great", "absolutely terrible", "it was okay"],
        "label": ["positive", "negative", "neutral"],
    })
    ds = _encode_dataset(df, tokenizer, max_length=16)
    assert len(ds) == 3
    assert set(ds.column_names) == {"input_ids", "attention_mask", "labels"}
    sample = ds[0]
    assert sample["input_ids"].shape[-1] == 16
    assert sample["labels"].item() in {0, 1, 2}


def test_encode_dataset_drops_invalid_labels(monkeypatch):
    _install_fake_datasets(monkeypatch)
    tokenizer = DummyTokenizer()
    df = pd.DataFrame({
        "text": ["a", "b", "c"],
        "label": ["positive", "garbage", "negative"],
    })
    ds = _encode_dataset(df, tokenizer, max_length=8)
    assert len(ds) == 2


def test_train_distilbert_rejects_missing_columns(tmp_path):
    df = pd.DataFrame({"text": ["a"], "rating": [5]})  # no label
    with pytest.raises(ValueError, match="text.*label"):
        train_distilbert(df, tmp_path / "out")
