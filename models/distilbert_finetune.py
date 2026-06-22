"""DistilBERT fine-tuning for sentiment classification.

Production defaults come from the winning `distilbert-final` run in
`notebooks/02_distilbert_tuning.ipynb` (`distilbert-baseline` config +
val threshold tuning).

CLI:
    python models/distilbert_finetune.py \\
        --data data/sample/reviews_sample.csv \\
        --out  models/artifacts/distilbert \\
        --epochs 4

MLflow behavior matches ``models/train.py`` — if ``MLFLOW_TRACKING_URI`` is set,
logging is required; if unset, only local artifacts are written.

Owner: Van (Modeler), Amelia (second pair).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.metrics import (  # noqa: E402
    INFERENCE_BATCH_SIZE,
    classification_metrics,
    measure_inference_latency_ms,
    mlflow_metrics,
    mlflow_training_params,
    split_metrics,
    train_metadata,
)
from models.splits import train_val_test_oot_split  # noqa: E402

LABELS = ["negative", "neutral", "positive"]
LABEL2ID = {lbl: i for i, lbl in enumerate(LABELS)}
ID2LABEL = {i: lbl for lbl, i in LABEL2ID.items()}
NEG_IDX = 0

DEFAULT_MODEL = "distilbert-base-uncased"
DEFAULT_MAX_LEN = 256
DEFAULT_NEG_THRESHOLD = 0.14
DEFAULT_MODEL_NAME = "sentiment-distilbert"
DEFAULT_EXPERIMENT = "sentiment-distilbert"
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
    mlflow_run_id: Optional[str] = None
    mlflow_model_version: Optional[str] = None


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


def _predict_labels(model, tokenizer, texts: List[str], cfg: TrainConfig) -> List[str]:
    """Thresholded predictions from an in-memory model (matches serving)."""
    import torch

    model.eval()
    with torch.no_grad():
        enc = tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.max_length,
        )
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=-1).numpy()
        base_pred = np.argmax(probs, axis=-1)
        ids = np.where(probs[:, NEG_IDX] >= cfg.neg_threshold, NEG_IDX, base_pred)
    return [ID2LABEL[int(i)] for i in ids]


def _evaluate_split_df(
    model,
    tokenizer,
    frame: pd.DataFrame,
    split: str,
    cfg: TrainConfig,
) -> Optional[dict]:
    if frame.empty:
        return None
    texts = frame["text"].astype(str).tolist()
    y_true = frame["label"].tolist()
    y_pred = _predict_labels(model, tokenizer, texts, cfg)
    return split_metrics(y_true, y_pred, split, labels=LABELS)


def _compute_metrics(eval_pred) -> dict:
    """Trainer metrics — ``f1_negative`` drives checkpoint selection (argmax, no threshold)."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    y_true = [ID2LABEL[int(i)] for i in labels]
    y_pred = [ID2LABEL[int(i)] for i in preds]
    metrics = classification_metrics(y_true, y_pred, labels=LABELS)
    metrics.pop("report", None)
    metrics.pop("f1_weighted", None)
    return metrics


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


def _log_to_mlflow(model: Any, out_dir: Path, cfg: TrainConfig, metrics: dict) -> tuple[str, Optional[str]]:
    """Log run + register model. Returns (run_id, model_version)."""
    import mlflow
    import mlflow.pytorch

    tracking_uri = os.environ["MLFLOW_TRACKING_URI"]
    model_name = os.getenv("MODEL_NAME", DEFAULT_MODEL_NAME)
    experiment = os.getenv("MLFLOW_EXPERIMENT", DEFAULT_EXPERIMENT)

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment)

    with mlflow.start_run() as run:
        mlflow.log_params({
            **mlflow_training_params(metrics),
            "model_type": "distilbert_sentiment",
            "base_model": cfg.base_model,
            "max_length": cfg.max_length,
            "num_epochs": cfg.num_epochs,
            "learning_rate": cfg.learning_rate,
            "batch_size": cfg.batch_size,
            "eval_batch_size": cfg.eval_batch_size,
            "weight_decay": cfg.weight_decay,
            "warmup_ratio": cfg.warmup_ratio,
        })
        mlflow.log_metrics(mlflow_metrics(metrics))
        mlflow.log_artifact(str(out_dir / SENTIMENT_CONFIG_FILE))

        info = mlflow.pytorch.log_model(
            pytorch_model=model,
            artifact_path="model",
            registered_model_name=model_name,
            code_paths=[str(ROOT / "models")],
        )
        version = getattr(info, "registered_model_version", None)
        return run.info.run_id, str(version) if version else None


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

    if not {"text", "label"}.issubset(df.columns):
        raise ValueError("df must have 'text' and 'label' columns")

    # Late imports — transformers+torch are heavy; only callers that need
    # this module pay the import cost.
    import torch
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        Trainer,
        TrainingArguments,
    )

    torch.manual_seed(cfg.seed)

    split = train_val_test_oot_split(
        df, oot_frac=0.2, val_frac=0.15, test_frac=test_size, seed=cfg.seed
    )
    trainer_eval = split.val if not split.val.empty else split.test

    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.base_model,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    train_ds = _encode_dataset(split.train, tokenizer, cfg.max_length)
    eval_ds = _encode_dataset(trainer_eval, tokenizer, cfg.max_length) if not trainer_eval.empty else None

    quick_run = cfg.max_steps > 0
    can_eval = eval_ds is not None and not quick_run
    args = TrainingArguments(
        output_dir=str(out_dir / "_trainer"),
        num_train_epochs=cfg.num_epochs,
        max_steps=cfg.max_steps,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.eval_batch_size,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        eval_strategy="epoch" if can_eval else "no",
        save_strategy="epoch" if can_eval else "no",
        load_best_model_at_end=can_eval,
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

    metrics: dict = {
        "neg_threshold": float(cfg.neg_threshold),
        "n_train": int(len(split.train)),
        "n_val": int(len(split.val)),
        "n_test": int(len(split.test)),
        "n_oot": int(len(split.oot)),
        "cutoff_date": None if split.cutoff_date is None else str(split.cutoff_date),
        "inference_batch_size": INFERENCE_BATCH_SIZE,
        **train_metadata(split.train),
    }

    for split_name, frame in (("val", split.val), ("test", split.test), ("oot", split.oot)):
        split_scores = _evaluate_split_df(model, tokenizer, frame, split_name, cfg)
        if split_scores:
            metrics.update(split_scores)

    if not split.test.empty:
        texts = split.test["text"].astype(str).tolist()
        metrics["inference_latency_ms"] = measure_inference_latency_ms(
            lambda batch: _predict_labels(model, tokenizer, batch, cfg),
            texts,
            batch_size=INFERENCE_BATCH_SIZE,
        )

    # Legacy unprefixed keys for callers expecting flat metrics.
    if "test_f1_macro" in metrics:
        metrics["f1_macro"] = metrics["test_f1_macro"]
        metrics["f1_negative"] = metrics["test_f1_negative"]
        metrics["f1_neg"] = metrics["test_f1_negative"]
        metrics["precision_neg"] = metrics["test_precision_negative"]
        metrics["recall_neg"] = metrics["test_recall_negative"]
        metrics["accuracy"] = metrics.get("test_accuracy", 0.0)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    _write_sentiment_config(out_dir, cfg, metrics)

    run_id: Optional[str] = None
    version: Optional[str] = None
    if os.getenv("MLFLOW_TRACKING_URI"):
        run_id, version = _log_to_mlflow(model, out_dir, cfg, metrics)

    metrics["mlflow_run_id"] = run_id
    metrics["mlflow_model_version"] = version
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
    for key in (
        "test_f1_negative", "val_f1_negative", "oot_f1_negative",
        "test_f1_macro", "val_f1_macro",
        "test_recall_negative", "test_precision_negative",
        "test_f1_positive", "test_f1_neutral",
        "inference_latency_ms", "mlflow_run_id", "mlflow_model_version",
    ):
        if key in metrics:
            print(f"{key}: {metrics[key]}")


if __name__ == "__main__":
    main()
