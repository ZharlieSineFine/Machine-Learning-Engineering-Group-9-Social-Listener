from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}

DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_MAX_LEN = 128


@dataclass
class TrainConfig:
    base_model: str = DEFAULT_MODEL
    max_length: int = DEFAULT_MAX_LEN
    num_epochs: int = 1
    max_steps: int = -1            # -1 means "use num_epochs", any positive int caps steps (test path)
    learning_rate: float = 5e-5
    batch_size: int = 16
    seed: int = 42


@dataclass
class TrainOutput:
    model_dir: str
    metrics: dict = field(default_factory=dict)


def _encode_dataset(df: pd.DataFrame, tokenizer, max_length: int):
    #Tokenize the text column and attach integer labels.
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
    #Trainer-friendly compute_metrics; surfaces negative-class metrics.
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
        "f1_neg": float(neg["f1-score"]),
        "precision_neg": float(neg["precision"]),
        "recall_neg": float(neg["recall"]),
    }


def train_distilbert(
    df: pd.DataFrame,
    out_dir: Path,
    cfg: TrainConfig | None = None,
    test_size: float = 0.2,
) -> Tuple[Path, dict]:
    #Fine-tune DistilBERT on df and save the model to out_dir; returns (out_dir, metrics_dict) with eval macro-F1 + accuracy.
    cfg = cfg or TrainConfig()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("df must have 'text' and 'label' columns")

    #Late imports: transformers+torch are heavy, so only callers that need this module pay the import cost.
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

    args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        eval_strategy="no",            # we evaluate manually after train (`evaluation_strategy` in older transformers)
        save_strategy="no",
        logging_steps=50,
        report_to=[],                  # don't auto-log to wandb/tensorboard
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
    metrics = trainer.evaluate()  # uses compute_metrics → accuracy + f1_macro

    # Save the fine-tuned model + tokenizer so the API can reload from this dir.
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    return out_dir, metrics


def predict(texts: List[str], model_dir: Path) -> List[str]:
    #Convenience predictor used by the smoke check in main.
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.eval()
    with torch.no_grad():
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=DEFAULT_MAX_LEN)
        logits = model(**enc).logits
        ids = logits.argmax(-1).tolist()
    return [ID2LABEL[i] for i in ids]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--batch-size", type=int, default=16)
    args = ap.parse_args()

    cfg = TrainConfig(
        num_epochs=args.epochs, max_steps=args.max_steps, batch_size=args.batch_size
    )
    df = pd.read_csv(args.data)
    out_dir, metrics = train_distilbert(df, args.out, cfg)
    print(f"saved to {out_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
