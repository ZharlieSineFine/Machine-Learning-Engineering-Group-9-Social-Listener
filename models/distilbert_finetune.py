"""DistilBERT fine-tuning for sentiment classification.

Production defaults come from the winning `distilbert-final` run in
`notebooks/02_distilbert_tuning.ipynb` (`distilbert-baseline` config +
val threshold tuning).

CLI:
    python models/distilbert_finetune.py \\
        --data data/sample/reviews_sample.csv \\
        --out  models/artifacts/distilbert \\
        --epochs 4

Owner: Van (Modeler), Amelia (second pair).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}
NEG_IDX = 0

DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_MAX_LEN = 256
DEFAULT_NEG_THRESHOLD = 0.14
SENTIMENT_CONFIG_FILE = "sentiment_config.json"


@dataclass
class TrainConfig:
    """Tuned defaults from notebook 02 (`distilbert-baseline`)."""

    base_model: str = DEFAULT_MODEL
    max_length: int = DEFAULT_MAX_LEN
    num_epochs: int = 4
    max_steps: int = -1            # >0 caps steps for fast test runs
    learning_rate: float = 2e-5
    batch_size: int = 16
    eval_batch_size: int = 32
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    neg_threshold: float = DEFAULT_NEG_THRESHOLD
    seed: int = 42


@dataclass
class TrainOutput:
    model_dir: str
    metrics: dict = field(default_factory=dict)


def _encode_dataset(df: pd.DataFrame, tokenizer, max_length: int):
    """Tokenize the text column and attach integer labels."""
    from datasets import Dataset

    df = df.dropna(subset=["text", "label"]).copy()
    df = df[df["label"].isin(LABEL2ID)]
    df["label_id"] = df["label"].map(LABEL2ID)
    ds = Dataset.from_pandas(df[["text", "label_id"]], preserve_index=False)

    def _tok(batch):
        enc = tokenizer(
            batch["text"], truncation=True, padding="max_length", max_length=max_length
        )
        enc["labels"] = batch["label_id"]
        return enc

    ds = ds.map(_tok, batched=True, remove_columns=["text", "label_id"])
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds


def _compute_metrics(eval_pred) -> dict:
    """Trainer metrics — `f1_negative` drives checkpoint selection."""
    from sklearn.metrics import accuracy_score, classification_report, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    report = classification_report(
        labels,
        preds,
        labels=list(LABEL2ID.values()),
        target_names=LABELS,
        output_dict=True,
        zero_division=0,
    )
    neg = report["negative"]
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "f1_macro": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "f1_negative": float(neg["f1-score"]),
        "f1_neg": float(neg["f1-score"]),
        "precision_neg": float(neg["precision"]),
        "recall_neg": float(neg["recall"]),
    }


def _write_sentiment_config(out_dir: Path, cfg: TrainConfig, metrics: dict) -> None:
    payload = {
        "neg_threshold": cfg.neg_threshold,
        "labels": LABELS,
        "train_config": asdict(cfg),
        "metrics": metrics,
    }
    (out_dir / SENTIMENT_CONFIG_FILE).write_text(json.dumps(payload, indent=2))


def load_neg_threshold(model_dir: Path) -> float:
    return load_sentiment_config(model_dir).get("neg_threshold", DEFAULT_NEG_THRESHOLD)


def load_sentiment_config(model_dir: Path) -> dict:
    cfg_path = Path(model_dir) / SENTIMENT_CONFIG_FILE
    if cfg_path.exists():
        return json.loads(cfg_path.read_text())
    return {"neg_threshold": DEFAULT_NEG_THRESHOLD, "train_config": {"max_length": DEFAULT_MAX_LEN}}


def train_distilbert(
    df: pd.DataFrame,
    out_dir: Path,
    cfg: TrainConfig | None = None,
    test_size: float = 0.2,
) -> Tuple[Path, dict]:
    """Fine-tune DistilBERT on `df` and save the model to `out_dir`.

    Returns (out_dir, metrics_dict). When not in quick-test mode (`max_steps <= 0`),
    loads the best epoch checkpoint by validation `f1_negative`.
    """
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

<<<<<<< HEAD
=======
    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("df must have 'text' and 'label' columns")

    # Late imports — transformers+torch are heavy; only callers that need
    # this module pay the import cost.
>>>>>>> origin/feature/full_flow
    import torch
    from sklearn.model_selection import train_test_split
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(cfg.seed)

    train_df, eval_df = train_test_split(
        df, test_size=test_size, random_state=cfg.seed, stratify=df["label"]
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = _encode_dataset(train_df, tokenizer, cfg.max_length)
    eval_ds = _encode_dataset(eval_df, tokenizer, cfg.max_length)

    quick_run = cfg.max_steps > 0
    args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="no" if quick_run else "epoch",
        save_strategy="no" if quick_run else "epoch",
        load_best_model_at_end=not quick_run,
        metric_for_best_model="f1_negative",
        greater_is_better=True,
        fp16=torch.cuda.is_available(),
        logging_steps=50,
        report_to=[],
        seed=cfg.seed,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        compute_metrics=_compute_metrics,
    )

    trainer.train()
    metrics = trainer.evaluate()
    metrics = {k.removeprefix("eval_"): v for k, v in metrics.items()}

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_sentiment_config(out_dir, cfg, metrics)

    return out_dir, metrics


def _predict_ids(
    texts: List[str],
    model_dir: Path,
    threshold: float | None = None,
) -> List[int]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    config = load_sentiment_config(model_dir)
    if threshold is None:
        threshold = float(config.get("neg_threshold", DEFAULT_NEG_THRESHOLD))
    max_length = int(config.get("train_config", {}).get("max_length", DEFAULT_MAX_LEN))

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()

    with torch.no_grad():
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).numpy()
        base_pred = np.argmax(probs, axis=-1)
        return np.where(probs[:, NEG_IDX] >= threshold, NEG_IDX, base_pred).tolist()


def predict(
    texts: List[str],
    model_dir: Path,
    threshold: float | None = None,
) -> List[str]:
    """Predict labels, applying the tuned negative threshold when set."""
    ids = _predict_ids(texts, model_dir, threshold=threshold)
    return [ID2LABEL[i] for i in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=TrainConfig.num_epochs)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=TrainConfig.batch_size)
    ap.add_argument("--neg-threshold", type=float, default=DEFAULT_NEG_THRESHOLD)
    args = ap.parse_args()

    cfg = TrainConfig(
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        neg_threshold=args.neg_threshold,
    )
    df = pd.read_csv(args.data)
    out_dir, metrics = train_distilbert(df, args.out, cfg)
    print(f"saved to {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
