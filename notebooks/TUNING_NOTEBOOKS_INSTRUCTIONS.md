# Sentiment Model Tuning Notebooks — Reference

Reference for the two offline experimentation notebooks in `notebooks/`:

- `notebooks/01_tfidf_logreg_tuning.ipynb`
- `notebooks/02_distilbert_tuning.ipynb`

These notebooks do **not** connect to Airflow, Postgres, or MinIO. They read from a local
CSV (built by `notebooks/2026-06-09_van_gold_50k_eda.ipynb`) and log experiments to MLflow.

**Prerequisites**

- `data/local/gold/gold_50k_training.csv` must exist (gitignored; columns: `text`, `label`, `review_date`).
- Notebook 01: `pandas`, `scikit-learn`, `matplotlib`, `mlflow`, `joblib`, `requests`.
- Notebook 02: above plus `transformers`, `torch`, and **`datasets`** (`pip install datasets` if missing).

---

## 0. Shared Setup (both notebooks)

### Data source

```python
from pathlib import Path

# Resolve repo root whether the kernel cwd is repo root or notebooks/
ROOT = Path.cwd()
if not (ROOT / "data" / "local").exists() and (ROOT.parent / "data" / "local").exists():
    ROOT = ROOT.parent

DATA_PATH = ROOT / "data" / "local" / "gold" / "gold_50k_training.csv"
ARTIFACT_DIR = ROOT / "models" / "artifacts"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
```

Expected columns: `text`, `label`, `review_date`  
Label values: `"negative"`, `"neutral"`, `"positive"`  
Label distribution (approximate): negative 21%, neutral 7%, positive 72%  
Total rows: **50,220** (Charlie's local gold export, 2021-05-02 → 2022-04-10)

### MLflow tracking

Both notebooks try the Docker MLflow server first (host port **5001**), then fall back to a file store:

```python
import mlflow
import requests

# docker-compose maps host 5001 -> container 5000 (5000 is often taken on macOS).
TRACKING_URI = "http://localhost:5001"
try:
    requests.get(TRACKING_URI, timeout=2)
    mlflow.set_tracking_uri(TRACKING_URI)
except requests.exceptions.RequestException:
    mlflow.set_tracking_uri(f"file:{(ROOT / 'mlruns').as_posix()}")
```

Open the UI at **`http://localhost:5001`** when `docker compose up` is running.
Without Docker, runs land in `./mlruns`; view them with
`mlflow ui --backend-store-uri file:./mlruns`.

| Notebook | Experiment name |
|---|---|
| 01 | `sentiment-tfidf-logreg` |
| 02 | `sentiment-distilbert` |

---

## 1. Data Splitting — Apply in Both Notebooks

Identical split strategy in both notebooks. Put this in a shared utility cell near the top
so it is easy to audit.

### Rules

- OOT and demo splits are **strictly temporal** — determined by `review_date` cutoff.
- Train / val / test are drawn from the remaining pool using **random shuffle + stratify on label** (80/10/10 of `rest_df`).
- The demo/replay split is **never used for training or evaluation** — saved to CSV for use by `data/ingest/replay.py` during the live demo.
- Demo CSV writes happen in **notebook 01 only** (notebook 02 documents that the writes are idempotent if re-run).

### Cutoff dates (adjusted to this export)

The original template used `2022-07-01` / `2022-10-01`, but the export ends at **2022-04-10**,
so those cutoffs would leave OOT and demo empty. Both notebooks use quantile-based cutoffs:

| Constant | Value | Role |
|---|---|---|
| `OOT_CUTOFF` | `"2021-12-11"` | Start of out-of-time holdout |
| `DEMO_CUTOFF` | `"2022-01-09"` | Start of demo/replay holdout |

### Implementation

```python
import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42

df = pd.read_csv(DATA_PATH, parse_dates=["review_date"])
df = df.sort_values("review_date").reset_index(drop=True)

# ── Temporal holdouts ──────────────────────────────────────────────────────────
OOT_CUTOFF = "2021-12-11"
DEMO_CUTOFF = "2022-01-09"

demo_df = df[df.review_date >= DEMO_CUTOFF].copy()
oot_df = df[(df.review_date >= OOT_CUTOFF) & (df.review_date < DEMO_CUTOFF)].copy()
rest_df = df[df.review_date < OOT_CUTOFF].copy()

# ── Random splits from rest_df ─────────────────────────────────────────────────
# 80 / 10 / 10 of rest_df (~86% of total) → approx 69/9/9 of total
rest_df = rest_df.sample(frac=1, random_state=SEED).reset_index(drop=True)

train_df, temp_df = train_test_split(
    rest_df, test_size=0.2, stratify=rest_df["label"], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=SEED
)

# ── Sanity check ───────────────────────────────────────────────────────────────
for name, split in [("train", train_df), ("val", val_df), ("test", test_df),
                    ("oot", oot_df), ("demo", demo_df)]:
    dist = split.label.value_counts(normalize=True).round(3).to_dict()
    print(f"{name:6s}  n={len(split):>6,}  {dist}")

# ── Save demo/replay files (notebook 01 only) ──────────────────────────────────
# Window A: representative clean sample
demo_df.sample(n=min(500, len(demo_df)), random_state=SEED).to_csv(
    ROOT / "data" / "sample" / "demo_window_a_clean.csv", index=False
)
# Window B: poisoned — oversample negatives to simulate brand crisis
neg = demo_df[demo_df.label == "negative"]
other_pool = demo_df[demo_df.label != "negative"]
other = other_pool.sample(n=min(100, len(other_pool)), random_state=SEED)
pd.concat([neg, other]).sample(frac=1, random_state=SEED).to_csv(
    ROOT / "data" / "sample" / "demo_window_b_poisoned.csv", index=False
)
print("Demo replay files saved.")
```

### Actual split sizes (from 50,220 total rows)

| Split | Rows | Share | How split |
|---|---|---|---|
| train | 34,413 | 68.5% | random stratified |
| val | 4,302 | 8.6% | random stratified |
| test | 4,302 | 8.6% | random stratified |
| oot | 5,147 | 10.2% | temporal (2021-12-11 → 2022-01-08) |
| demo | 2,056 | 4.1% | temporal (2022-01-09 → 2022-04-10), saved to CSV only |

Label proportions are stable across train/val/test (~72% positive, ~21% negative, ~7% neutral).

---

## 2. Primary Metrics — Apply in Both Notebooks

The project goal is **social listening**: detecting surges in negative sentiment.
Metric priority order:

| Priority | Metric | Reason |
|---|---|---|
| 1 | `f1_negative` | Primary model selection metric. Balances catching negatives with not over-flagging. |
| 2 | `recall_negative` | Missing a real negative (false negative) is the costliest error. |
| 3 | `precision_negative` | Logged for completeness; used to diagnose if model over-flags. |
| 4 | `f1_macro` | Reference metric for academic/report comparison across models. |
| 5 | `f1_neutral` | Logged but excluded from model selection gate. |

### Shared evaluation function — utility cell in both notebooks

```python
from sklearn.metrics import (
    f1_score, recall_score, precision_score, classification_report
)

LABELS   = ["negative", "neutral", "positive"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
NEG_IDX  = 0  # index of "negative" in LABELS list

def evaluate(y_true, y_pred, split_name: str, log_to_mlflow: bool = True) -> dict:
    """Compute and optionally log all metrics for a given split."""
    metrics = {
        f"{split_name}_f1_negative":        f1_score(y_true, y_pred, labels=[NEG_IDX], average="macro"),
        f"{split_name}_recall_negative":    recall_score(y_true, y_pred, labels=[NEG_IDX], average="macro"),
        f"{split_name}_precision_negative": precision_score(y_true, y_pred, labels=[NEG_IDX], average="macro"),
        f"{split_name}_f1_macro":           f1_score(y_true, y_pred, average="macro"),
        f"{split_name}_f1_neutral":         f1_score(y_true, y_pred, labels=[1], average="macro"),
        f"{split_name}_f1_positive":        f1_score(y_true, y_pred, labels=[2], average="macro"),
    }
    print(f"\n── {split_name.upper()} ──")
    print(classification_report(y_true, y_pred, target_names=LABELS))
    if log_to_mlflow:
        mlflow.log_metrics(metrics)
    return metrics

# Call on val, test, and OOT for every experiment run.
```

Notebook 02 also defines `ID2LABEL = {i: l for l, i in LABEL2ID.items()}` for HuggingFace.

---

## 3. Notebook 1 — `01_tfidf_logreg_tuning.ipynb`

**MLflow experiment:** `sentiment-tfidf-logreg`  
**Purpose:** Establish a strong sklearn baseline → `sentiment-baseline` in the MLflow Registry
(thin-slice-first rule from WORKFLOW.md).

### Section structure

```
[0] Imports and config (ROOT, DATA_PATH, ARTIFACT_DIR, SEED, MLflow)
[1] Data loading and splits + demo replay CSV export
    └── Shared evaluation function
[2] Preprocessing (clean_text + label encoding)
[3] Experiment A — Baseline (no imbalance handling)
[4] Experiment B — Class weights
[5] Experiment C — Oversample neutral (manual resample to 8K)
[6] Experiment D — Hyperparameter tuning (GridSearchCV, 72 candidates)
[7] Threshold tuning on best model (val only; saves PNG)
[8] Final evaluation on test + OOT (best config + threshold)
[9] Register best model to MLflow
```

### Section 2 — Preprocessing

```python
import re

def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+", "", text)          # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)    # keep alphanumeric
    text = re.sub(r"\s+", " ", text).strip()
    return text

train_df["clean_text"] = train_df["text"].map(clean_text)
val_df["clean_text"]   = val_df["text"].map(clean_text)
test_df["clean_text"]  = test_df["text"].map(clean_text)
oot_df["clean_text"]   = oot_df["text"].map(clean_text)

y_train = train_df["label"].map(LABEL2ID).values
y_val   = val_df["label"].map(LABEL2ID).values
y_test  = test_df["label"].map(LABEL2ID).values
y_oot   = oot_df["label"].map(LABEL2ID).values
```

### Section 3 — Experiment A: Baseline

Run name: `logreg-baseline`. TF-IDF `(1,2)`, `max_features=50000`, LogReg `C=1.0`, no class weight.

### Section 4 — Experiment B: Class weights

Run name: `logreg-class-weights`. Same TF-IDF; `class_weight="balanced"`.

### Section 5 — Experiment C: Oversample neutral

Run name: `logreg-oversample-neutral`. Neutral upsampled to 8K in **train only** (via
`sklearn.utils.resample`), plus `class_weight="balanced"`.

### Section 6 — Experiment D: GridSearchCV

Run name: `logreg-gridsearch`. Scored on `f1_negative` via `PredefinedSplit` (train = fold -1,
val = fold 0). **72 candidates × 1 fold** — expect this to take a while on CPU.

```python
param_grid = {
    "tfidf__ngram_range":  [(1, 1), (1, 2), (1, 3)],
    "tfidf__max_features": [30000, 50000, 100000],
    "clf__C":              [0.1, 1.0, 5.0, 10.0],
    "clf__class_weight":   ["balanced", None],
}
```

Best estimator stored as `best_logreg_pipe`. GridSearch params logged with `str(v)` for
MLflow compatibility.

### Section 7 — Threshold tuning

Tuned on **val** only. If `P(negative) >= t`, predict negative; otherwise keep argmax.
Plot saved to `notebooks/logreg_threshold_curve.png` (not logged to MLflow until the final run).

```python
val_probs_neg = best_logreg_pipe.predict_proba(val_df["clean_text"])[:, NEG_IDX]
val_base_preds = best_logreg_pipe.predict(val_df["clean_text"])

thresholds = np.arange(0.10, 0.70, 0.02)
# ... sweep t, pick t that maximises val f1_negative → best_threshold
threshold_plot_path = ROOT / "notebooks" / "logreg_threshold_curve.png"
```

### Section 8 — Final evaluation

Run name: `logreg-final`. Applies `best_threshold` to test and OOT. Stores `final_run_id`
for registration in the next section.

```python
def predict_with_threshold(pipe, texts, threshold):
    probs = pipe.predict_proba(texts)[:, NEG_IDX]
    base  = pipe.predict(texts)
    return np.where(probs >= threshold, NEG_IDX, base)

with mlflow.start_run(run_name="logreg-final"):
    mlflow.log_params({**{k: str(v) for k, v in gs.best_params_.items()},
                       "neg_threshold": best_threshold})
    mlflow.log_artifact(str(threshold_plot_path))
    # evaluate test + OOT ...
    final_run_id = mlflow.active_run().info.run_id
```

### Section 9 — Register best model

```python
import joblib

joblib.dump(best_logreg_pipe, ARTIFACT_DIR / "tfidf_logreg_best.pkl")

with mlflow.start_run(run_id=final_run_id):
    mlflow.sklearn.log_model(
        best_logreg_pipe,
        artifact_path="model",
        registered_model_name="sentiment-baseline",
    )
```

---

## 4. Notebook 2 — `02_distilbert_tuning.ipynb`

**MLflow experiment:** `sentiment-distilbert`  
**Purpose:** Fine-tune DistilBERT; compare against the sklearn baseline.
Registered as `sentiment-distilbert` in MLflow Registry.

> **Hardware note:** the project venv may have CPU-only torch. 4 epochs × ~34K rows on CPU
> takes many hours. Use a GPU runtime (e.g. Colab with the CSV uploaded) or subsample /
> set `EPOCHS = 1` for a wiring check first.

### Section structure

```
[0] Imports and config (MODEL_CHECKPOINT, USE_GPU, OUTPUT_DIR, …)
[1] Data loading and splits (same cutoffs/seeds as notebook 01; no demo CSV write)
    └── Shared evaluation function + y_train/y_val/y_test/y_oot encoding
[2] Tokenization (encode_df → HuggingFace Dataset)
[3] Experiment A — Baseline fine-tune (no imbalance handling)
[4] Experiment B — Weighted loss (inverse-frequency class weights)
[5] Experiment C — Weighted loss + neutral oversample (8K)
[6] Threshold tuning on best model (val only; saves PNG)
[7] Final evaluation on test + OOT
[8] Register best model to MLflow
[9] LR sensitivity experiment (optional — 3 single-epoch probes)
    └── Model selection rule (markdown summary)
```

### Section 0 — Config

```python
MODEL_CHECKPOINT = "distilbert-base-uncased"
MAX_LENGTH       = 256
BATCH_TRAIN      = 16    # reduce to 8 on CPU
BATCH_EVAL       = 32
EPOCHS           = 4
LR               = 2e-5
WEIGHT_DECAY     = 0.01
WARMUP_RATIO     = 0.1
OUTPUT_DIR       = str(ROOT / "checkpoints" / "distilbert-sentiment")
SEED             = 42
USE_GPU          = torch.cuda.is_available()
```

### Section 2 — Tokenization

```python
from datasets import Dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_CHECKPOINT)

def encode_df(frame: pd.DataFrame) -> Dataset:
    ds = Dataset.from_pandas(
        frame[["text", "label"]].assign(label=frame["label"].map(LABEL2ID)),
        preserve_index=False,
    )
    ds = ds.map(
        lambda b: tokenizer(b["text"], truncation=True,
                            padding="max_length", max_length=MAX_LENGTH),
        batched=True,
    )
    ds = ds.rename_column("label", "labels")
    ds.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
    return ds

train_ds = encode_df(train_df)
val_ds   = encode_df(val_df)
test_ds  = encode_df(test_df)
oot_ds   = encode_df(oot_df)
```

### Section 3 — Shared training helpers + Experiment A

`get_training_args()` uses `fp16=USE_GPU` (mixed precision only when CUDA is available).
`metric_for_best_model="f1_negative"` drives checkpoint selection.

Run name: `distilbert-baseline`. Standard `Trainer`, no class weighting.

### Section 4 — Experiment B: Weighted loss

Run name: `distilbert-weighted-loss`. Custom `WeightedLossTrainer` with inverse-frequency
`CrossEntropyLoss` weights. Accepts `**kwargs` in `compute_loss` for newer transformers versions.

### Section 5 — Experiment C: Weighted loss + oversample neutral

Run name: `distilbert-weighted-loss-oversample`. Neutral upsampled to 8K before tokenizing;
uses `WeightedLossTrainer`.

### Section 6 — Threshold tuning

Pick `best_trainer` by comparing `val_f1_negative` across Experiments A–C in MLflow UI
(default placeholder: `weighted_trainer`). Same threshold sweep as notebook 01.
Plot saved to `notebooks/distilbert_threshold_curve.png`.

### Section 7 — Final evaluation

Run name: `distilbert-final`. Logs threshold plot artifact; stores `final_run_id`.

```python
def predict_distilbert_with_threshold(trainer, dataset, threshold):
    logits    = trainer.predict(dataset).predictions
    probs     = torch.softmax(torch.tensor(logits), dim=-1).numpy()
    base_pred = np.argmax(probs, axis=-1)
    return np.where(probs[:, NEG_IDX] >= threshold, NEG_IDX, base_pred)
```

### Section 8 — Register best model

```python
best_dir = ARTIFACT_DIR / "distilbert_best"
best_trainer.save_model(str(best_dir))
tokenizer.save_pretrained(str(best_dir))

with mlflow.start_run(run_id=final_run_id):
    mlflow.pytorch.log_model(
        best_trainer.model,
        artifact_path="model",
        registered_model_name="sentiment-distilbert",
    )
```

### Section 9 — LR sensitivity (optional)

Three single-epoch probes logged as `distilbert-lr-probe-{lr}` for `lr in [5e-6, 2e-5, 5e-5]`.
Uses `WeightedLossTrainer` on `train_ds`.

---

## 5. Summary of MLflow Runs

| Notebook | Run name | Key variation |
|---|---|---|
| 01 | `logreg-baseline` | No imbalance handling |
| 01 | `logreg-class-weights` | `class_weight="balanced"` |
| 01 | `logreg-oversample-neutral` | Neutral oversampled to 8K |
| 01 | `logreg-gridsearch` | GridSearchCV on f1_negative (72 candidates) |
| 01 | `logreg-final` | Best config + best threshold |
| 02 | `distilbert-baseline` | No imbalance handling |
| 02 | `distilbert-weighted-loss` | Inverse-freq class weights |
| 02 | `distilbert-weighted-loss-oversample` | Weighted loss + neutral 8K |
| 02 | `distilbert-lr-probe-{lr}` | LR sensitivity (3 runs, optional) |
| 02 | `distilbert-final` | Best config + best threshold |

### Model selection rule

Pick the run with the highest **`test_f1_negative`**. If two runs are within 0.01 of each other,
prefer the simpler model (logreg over distilbert; no sampling over sampling).

### Registered model names (MLflow Registry)

| Name | Source | Artifact path |
|---|---|---|
| `sentiment-baseline` | Notebook 01 | `models/artifacts/tfidf_logreg_best.pkl` |
| `sentiment-distilbert` | Notebook 02 | `models/artifacts/distilbert_best/` (model + tokenizer) |

Both must be manually promoted to **`Staging`** in the MLflow UI before the FastAPI shadow
deploy can load them.

### Demo replay outputs (notebook 01)

| File | Purpose |
|---|---|
| `data/sample/demo_window_a_clean.csv` | 500-row representative clean window |
| `data/sample/demo_window_b_poisoned.csv` | All demo negatives + 100 non-negatives (brand-crisis sim) |

These live under a **tracked** folder — decide with the team whether to commit them or
gitignore them once generated.
