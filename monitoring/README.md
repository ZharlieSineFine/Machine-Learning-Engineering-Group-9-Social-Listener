# Monitoring — Drift & Performance

**Owner:** Charlie + Ha

Evidently runs inside Airflow as a Python lib (no standalone service). It lives in
two places, both backed by `monitoring/drift_checks.py`:

- **`evaluate_and_monitor`** (every 6h) — observational drift on live data; writes
  a report and, when drift blocks the gate, **triggers a retrain** of
  `medallion_train_cycle`.
- **`medallion_train_cycle` → `gate`** task — the promotion gate; runs between
  `train` and `promote` and **blocks promotion** of a regressed model.

## What we monitor

| Check | Source | Threshold | Action on fail |
|---|---|---|---|
| Data drift (text-length, rating, source mix) | Training frame vs. recent current window | per-column **PSI** ≥ `DRIFT_PSI_THRESHOLD` (0.2); blocks when share of drifted columns ≥ `DRIFT_THRESHOLD` (0.3) | Flag in dashboard / fire retrain; in the cycle gate, block promotion |
| Target drift (label distribution) | Training labels vs. rating-derived labels on the current window | label **PSI** ≥ `DRIFT_PSI_THRESHOLD` (0.2) | Flag in dashboard / fire retrain |
| Prediction-distribution drift (predicted-label mix) | Model scored on reference vs. predicted labels on the current window | prediction **PSI** ≥ `DRIFT_PSI_THRESHOLD` (0.2) | Flag in dashboard / fire retrain |
| Performance — macro-F1 | Model scored on reference vs. current | F1-macro drop > `DRIFT_F1_DROP_THRESHOLD` (3%) | **Block** `Staging → Production` |
| Performance — negative-class recall | Model scored on reference vs. current | recall_neg drop > `DRIFT_RECALL_NEG_DROP_THRESHOLD` (5%) | **Block** `Staging → Production` |

PSI (Population Stability Index) is the stattest applied to every column — data
features, the target label, and the prediction column. Standard bands: **< 0.1
no shift, 0.1–0.2 moderate, > 0.2 significant**. The observational monitor
(`run_monitor_drift`) emits one combined Evidently report covering all three
drift types; `MonitorResult.blocked` (data OR target OR prediction drift) drives
the retrain trigger.

**Current-window source (both/fallback):** the observational monitor prefers the
`predictions` table (logged production predictions over the last
`DRIFT_PRED_WINDOW_DAYS`, joined to `reviews` for rating/source) and falls back
to scoring the Production model on the recent silver window when the table is
empty.

The gate (`evaluate()`) blocks when **data drift OR either performance metric**
regresses. The negative class is treated as business-critical (catching negative
brand sentiment matters most), so it has its own recall guard alongside F1. The
gate's uploaded report **also covers target + prediction-distribution drift
(PSI)** — same shape as the observational monitor — but those are informational
and do **not** change the promote/block decision (that stays data drift +
measured performance). The model is scored once per side and those predictions
are reused for both the performance metrics and the prediction-drift column.

## How the gate works (`evaluate`)

1. `compute_drift(reference, current)` — Evidently `DataDriftPreset` (+
   `TargetDriftPreset` when a `label` column is present), restricted to the
   monitored columns shared by both frames. Free text is excluded.
2. If a `model` is given, score it **once per side** to get macro-F1 and
   negative-class recall (one `predict` call each — never two, so stateful models
   compare correctly).
3. Upload the HTML report to MinIO and append the `monitoring_reports` row
   **before** any block is raised, so a failing report stays in the dashboard.
4. Return `blocked_promotion`; `promote` honours it. With `raise_on_block=True`
   the task fails red via `PromotionBlocked`.

## Retrain loop

`evaluate_and_monitor` ends with `should_retrain` (a `ShortCircuitOperator` reading
the drift result) → `trigger_retrain` (a `TriggerDagRunOperator` firing
`medallion_train_cycle`). So scheduled drift detection on live data automatically
kicks off a fresh ingest→train→gate→promote cycle.

## Outputs

- HTML report uploaded to MinIO under `s3://monitoring/{YYYY-MM-DD}/<type>.html`
  (`data_drift` for the observational DAG, `performance` for the cycle gate)
- One row appended to Postgres `monitoring_reports` (run_date, report_type,
  report_url, drift_score, blocked_promotion)
- The dashboard reads the latest row and embeds the HTML

## Config (env)

| Var | Default | Meaning |
|---|---|---|
| `DRIFT_THRESHOLD` | `0.3` | Share of drifted columns that blocks |
| `DRIFT_STATTEST` | `psi` | Evidently per-column stattest |
| `DRIFT_PSI_THRESHOLD` | `0.2` | PSI value at/above which a column is "drifted" |
| `DRIFT_F1_DROP_THRESHOLD` | `0.03` | Macro-F1 drop that blocks |
| `DRIFT_RECALL_NEG_DROP_THRESHOLD` | `0.05` | Negative-class recall drop that blocks |
| `DRIFT_RECENT_PARTITIONS` | `7` | How many recent silver `review_date=` partitions form `current` |
| `DRIFT_PRED_WINDOW_DAYS` | `7` | Days of logged predictions forming `current` when the predictions table is used |
| `MODEL_NAME` / `MODEL_STAGE` | — / `Production` | MLflow model the monitor scores for prediction drift (pickle fallback otherwise) |
| `DRIFT_SILVER_ROOT` / `DRIFT_SAMPLE_CSV` | container/repo defaults | Override input locations |

When no silver partitions exist yet, the observational check degrades to
train-vs-itself (zero drift, gate passes) so the DAG stays green while wiring settles.
