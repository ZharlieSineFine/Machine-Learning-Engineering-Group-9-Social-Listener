#Slow but real DistilBERT fine-tuning smoke test.

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from models.distilbert_finetune import LABELS, TrainConfig, predict, train_distilbert


@pytest.mark.slow
def test_distilbert_tiny_training_e2e(tmp_path: Path):
    # 30 rows — small enough for a 2-step training run on CPU.
    rows = []
    for i in range(30):
        label = ["positive", "neutral", "negative"][i % 3]
        text = {
            "positive": "amazing food, will return",
            "neutral":  "it was okay, nothing special",
            "negative": "terrible service, would not recommend",
        }[label] + f" #{i}"
        rows.append({"text": text, "label": label})
    df = pd.DataFrame(rows)

    out_dir = tmp_path / "distilbert_out"
    cfg = TrainConfig(num_epochs=1, max_steps=2, batch_size=4, max_length=32)
    saved_dir, metrics = train_distilbert(df, out_dir, cfg)

    assert saved_dir == out_dir
    assert (saved_dir / "config.json").exists()
    assert (saved_dir / "tokenizer.json").exists() or (saved_dir / "tokenizer_config.json").exists()
    assert (saved_dir / "metrics.json").exists()

    assert "eval_accuracy" in metrics or "accuracy" in metrics
    f1_key = next(k for k in metrics if "f1_macro" in k)
    assert 0.0 <= metrics[f1_key] <= 1.0

    # Reload + predict.
    preds = predict(["the food was incredible", "absolutely awful"], saved_dir)
    assert len(preds) == 2
    assert all(p in LABELS for p in preds)
