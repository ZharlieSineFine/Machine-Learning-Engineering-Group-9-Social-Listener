"""Unit tests for models/distilbert_finetune.py — fast, no model download.

We verify the things we can verify without actually fine-tuning DistilBERT:
  * label maps are consistent (LABEL2ID <-> ID2LABEL roundtrip).
  * `_encode_dataset` produces torch-formatted tensors of the right shape.
  * `_compute_metrics` returns macro-F1 + accuracy in [0, 1].
  * `train_distilbert` rejects DataFrames missing required columns.
  * `TrainConfig` defaults match what the DAG expects.

The end-to-end "does it actually train" test is `tests/test_distilbert_slow.py`
and is gated by `RUN_SLOW=1` (or `pytest -m slow`).
"""
from __future__ import annotations

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
    assert cfg.num_epochs == 4
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
    assert set(metrics) >= {
        "accuracy", "f1_macro", "f1_negative", "f1_neg", "precision_neg", "recall_neg",
    }
    assert metrics["accuracy"] == 1.0  # all argmaxes match
    assert 0.0 <= metrics["f1_macro"] <= 1.0
    assert metrics["recall_neg"] == 1.0


def test_encode_dataset_returns_torch_tensors():
    """Use a real (tiny) tokenizer — no model weights download. Fast (<5s)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
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


def test_encode_dataset_drops_invalid_labels():
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
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
