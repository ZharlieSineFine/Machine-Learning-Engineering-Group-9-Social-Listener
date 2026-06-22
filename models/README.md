# Models — Training & Experimentation

**Owner:** Van (lead), Amelia (pair)

All training code lives here. Notebooks are for exploration only and don't run in CI.

## Files

| File | Phase | Purpose |
|---|---|---|
| `baseline_sklearn.py` | 1+ | TF-IDF + LogReg — tuned defaults from notebook 01 (`logreg-final`) |
| `distilbert_finetune.py` | 2 | DistilBERT fine-tune — tuned defaults from notebook 02 (`distilbert-final`) |
| `splits.py` | 1+ | Silver OOT split + gold train/val/test/oot/demo contract |
| `train.py` | 1+ | Sklearn entrypoint for `train_model` DAG; logs to MLflow, registers `sentiment-baseline` |
| `embeddings.py` | 2 | Gold-layer MiniLM embeddings (`all-MiniLM-L6-v2`, 384-d) |
| `topic_model.py` | Stretch | BERTopic on negative reviews |

## Train / validation / test / OOT split (`splits.py`)

Two split helpers share this module:

### Silver / sample CSV — `train_val_test_oot_split(df)`

`train_val_test_oot_split(df)` holds out the most recent reviews (by Silver `date`) as an
**out-of-time (OOT)** set, then splits the older in-time pool — stratified on `label` —
into train / validation / test:

- **train** — fit the model.
- **validation** — tune / select (threshold tuning uses this in notebook 01).
- **test** — in-time generalisation estimate (same period as train).
- **oot** — temporal generalisation estimate (genuinely later reviews).

`baseline_sklearn.train()` returns both: `f1_macro` / `f1_weighted` are the in-time **test**
scores; `f1_macro_oot` / `f1_weighted_oot` are the **OOT** scores. On the date-less seed CSV
the split degrades to a plain stratified train/val/test (OOT empty). Defaults: `oot_frac=0.2`,
`val_frac=0.15`, `test_frac=0.20`.

### Gold export — `split_gold(df)`

When `reviews_gold` is ready, the training export must include:

| Column | Type | Notes |
|---|---|---|
| `text` | string | Review body |
| `label` | string | `negative` \| `neutral` \| `positive` |
| `review_date` | datetime | Required for temporal OOT/demo splits |

```python
from models.splits import split_gold

splits = split_gold(gold_df)
train_df = splits.train   # fit here
val_df   = splits.val     # threshold tuning
test_df  = splits.test    # offline metrics
oot_df   = splits.oot     # temporal generalization check
# splits.demo — replay simulator only; never train on this
```

**Cutoffs** (from 50k export): `OOT_CUTOFF = "2021-12-11"`, `DEMO_CUTOFF = "2022-01-09"`.
Adjust in `splits.py` if the production gold date range changes.

## MLflow registry & shadow deploy

| Registered name | Tuned run | Promote to | Role |
|---|---|---|---|
| `sentiment-baseline` | `logreg-final` (v3) | **Production** | Serves `/predict` responses |
| `sentiment-distilbert` | `distilbert-final` (v1) | **Staging** | Shadow candidate (logged, not served) |

Compare runs in `notebooks/03_mlflow_model_comparison.ipynb` or `scripts/compare_mlflow_models.py`.

## MLflow conventions

- **Experiment name:** `sentiment-tfidf-logreg`, `sentiment-distilbert` (notebooks); `sentiment-baseline` (DAG stub)
- **Required metrics:** split-prefixed keys from `models/metrics.py` — `test_f1_negative`, `val_f1_negative`, `oot_f1_negative`, `test_f1_macro`, `val_f1_macro`, per-class `test_f1_positive` / `test_f1_neutral`, plus `inference_latency_ms`
- **Required params:** `neg_threshold`, `training_data_size`, `class_distribution`
- **Registered model names:** `sentiment-baseline`, `sentiment-distilbert`

If a metric name changes, update Evidently config and dashboard queries in the same PR.

## `embeddings.py` — separate from sentiment training

`embeddings.py` is **not** used by `train.py`, `baseline_sklearn.py`, or `distilbert_finetune.py`.
Those models embed text inline (TF-IDF or DistilBERT tokenizer).

| | `embeddings.py` | Sentiment models |
|---|---|---|
| **Purpose** | Populate `reviews_gold.embedding` in the medallion pipeline | Classify sentiment |
| **Called by** | `build_gold` DAG (Charlie/Ha) | `train_model` DAG → `train.py` / `distilbert_finetune.py` |
| **Implementation** | `sentence-transformers/all-MiniLM-L6-v2` (384-d) | Tuned sklearn + DistilBERT |
| **Owner** | Van implements; Charlie/Ha wire into `build_gold` | Van |

```python
from models.embeddings import embed, EMBEDDING_DIM

vectors = embed(review_texts)  # shape (n, 384), float32
```

**Environment variables:**

| Var | Default | Purpose |
|---|---|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Override model id |
| `EMBEDDING_STUB=1` | off | Deterministic stub vectors (CI/tests, no download) |

Charlie/Ha: ensure the `build_gold` runtime installs `models/requirements.txt`
(includes `sentence-transformers`). Downstream uses: Evidently drift, optional BERTopic.

## Phase 1 smoke

`train.py` should finish in < 60s on `data/sample/reviews_sample.csv` (scaffold data, no `date` column).
