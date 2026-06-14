# Models — Training & Experimentation

**Owner:** Van (lead), Amelia (pair)

All training code lives here. Notebooks are for exploration only and don't run in CI.

## Files

| File | Phase | Purpose |
|---|---|---|
| `baseline_sklearn.py` | 1+ | TF-IDF + LogReg — tuned defaults from notebook 01 (`logreg-final`) |
| `distilbert_finetune.py` | 2 | DistilBERT fine-tune — tuned defaults from notebook 02 (`distilbert-final`) |
| `splits.py` | 2 | Gold train/val/test/oot/demo split contract (handoff to Charlie/Ha) |
| `train.py` | 1+ | Sklearn entrypoint for `train_model` DAG; logs to MLflow, registers `sentiment-baseline` |
| `embeddings.py` | 2 | Gold-layer MiniLM embeddings (`all-MiniLM-L6-v2`, 384-d) |
| `topic_model.py` | Stretch | BERTopic on negative reviews |

## Gold handoff contract (Van → Charlie/Ha)

When `reviews_gold` is ready, the training export must include:

| Column | Type | Notes |
|---|---|---|
| `text` | string | Review body |
| `label` | string | `negative` \| `neutral` \| `positive` |
| `review_date` | datetime | Required for temporal OOT/demo splits |

Charlie/Ha: export gold to CSV or pass a DataFrame to training code.  
Van: call `models.splits.split_gold(df)` before fitting — same logic as notebooks 01/02.

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

`train.py` still uses `reviews_sample.csv` + a simple 80/20 split until the DAG passes gold data.
`distilbert_finetune.py` is a separate CLI entrypoint (not called from `train.py`).

## MLflow registry & shadow deploy

| Registered name | Tuned run | Promote to | Role |
|---|---|---|---|
| `sentiment-baseline` | `logreg-final` (v3) | **Production** | Serves `/predict` responses |
| `sentiment-distilbert` | `distilbert-final` (v1) | **Staging** | Shadow candidate (logged, not served) |

Compare runs in `notebooks/03_mlflow_model_comparison.ipynb` or `scripts/compare_mlflow_models.py`.

## MLflow conventions

- **Experiment name:** `sentiment-tfidf-logreg`, `sentiment-distilbert` (notebooks); `sentiment-baseline` (DAG stub)
- **Required metrics:** `f1_macro`, `f1_neg`, `precision_neg`, `recall_neg`, `accuracy`
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

`train.py` should finish in < 60s on `data/sample/reviews_sample.csv` (scaffold data, no `review_date`).
