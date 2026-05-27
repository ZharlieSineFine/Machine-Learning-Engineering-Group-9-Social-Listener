# Models — Training & Experimentation

**Owner:** Van (lead), Amelia (pair)

All training code lives here. Notebooks are for exploration only and don't run in CI.

## Files

| File | Phase | Purpose |
|---|---|---|
| `baseline_sklearn.py` | 1 | TF-IDF + LogisticRegression baseline |
| `train.py` | 1+ | Entrypoint called from the Airflow `train_model` DAG; logs to MLflow, registers best run |
| `evaluate.py` | 1+ | Held-out scoring; returns metric dict |
| `distilbert_finetune.py` | 2 | HuggingFace Trainer fine-tune of DistilBERT |
| `topic_model.py` | Stretch | BERTopic on negative reviews |

## MLflow conventions

- **Experiment name:** `sentiment-{branch_or_phase}`
- **Run name:** `{model}-{YYYYMMDD-HHMM}`
- **Required metrics:** `f1_macro`, `f1_neg`, `precision_neg`, `recall_neg`, `accuracy`
- **Required tags:** `model_family`, `dataset_version`, `git_sha`
- **Registered model name:** `sentiment-baseline` (phase 1), `sentiment-distilbert` (phase 2)

If a metric name changes, update the Evidently config and the dashboard query in the same PR.

## Phase 1 stub
`train.py` should be runnable as `python -m models.train --quick` and finish in < 60s on the sample dataset.
