# Models — Training & Experimentation

All training code lives here. Notebooks are for exploration only and don't run in CI.

## Files

| File | Phase | Purpose |
|---|---|---|
| `splits.py` | 1+ | Train / validation / test / **OOT** split keyed on the Silver `date` column |
| `baseline_sklearn.py` | 1 | TF-IDF + LogisticRegression baseline (fits on the OOT split) |
| `train.py` | 1+ | Entrypoint called from the Airflow `train_model` DAG; logs to MLflow, registers best run |
| `evaluate.py` | 1+ | Held-out scoring; returns metric dict |
| `distilbert_finetune.py` | 2 | HuggingFace Trainer fine-tune of DistilBERT |
| `topic_model.py` | Stretch | BERTopic on negative reviews |

## Train / validation / test / OOT split (`splits.py`)

`train_val_test_oot_split(df)` holds out the most recent reviews (by Silver `date`) as an
**out-of-time (OOT)** set, then splits the older "in-time" pool — stratified on `label` —
into train / validation / test:

- **train** — fit the model.
- **validation** — tune / select (the baseline doesn't tune yet, so it's reserved).
- **test** — in-time generalisation estimate (same period as train).
- **oot** — temporal generalisation estimate (genuinely later reviews).

`baseline_sklearn.train()` returns both: `f1_macro` / `f1_weighted` are the in-time **test**
scores (backward-compatible keys); `f1_macro_oot` / `f1_weighted_oot` are the **OOT** scores.
The **test → OOT gap** is the headline temporal-drift signal. On the date-less seed CSV the
split degrades to a plain stratified train/val/test (OOT empty). Defaults: `oot_frac=0.2`
(of dated rows), `val_frac=0.15`, `test_frac=0.20` (of the in-time pool).

## MLflow conventions

- **Experiment name:** `sentiment-{branch_or_phase}`
- **Run name:** `{model}-{YYYYMMDD-HHMM}`
- **Required metrics:** `f1_macro`, `f1_neg`, `precision_neg`, `recall_neg`, `accuracy`
- **Required tags:** `model_family`, `dataset_version`, `git_sha`
- **Registered model name:** `sentiment-baseline` (phase 1), `sentiment-distilbert` (phase 2)

If a metric name changes, update the Evidently config and the dashboard query in the same PR.
