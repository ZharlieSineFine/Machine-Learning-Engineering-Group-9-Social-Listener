# Monitoring — Drift & Performance

Evidently runs inside Airflow as a Python lib (no standalone service). It lives in
two places, both backed by `monitoring/drift_checks.py`:

- **`evaluate_and_monitor`** (per batch, data-triggered off `batch_inference`) —
  observational drift on live data; a read-only observer that writes a report and,
  when drift blocks the gate, **fires an alert** (no auto-retrain — see below).
- **`medallion_pipeline` → `gate`** task — the promotion gate; runs between
  `train` and `promote` and **blocks promotion** of a regressed model.

## What we monitor

Drift is detected **per column** with the Population Stability Index (PSI). Each
monitored column gets its own PSI test, and a single column over the threshold is
enough to flag drift — there is no share-of-columns gate. PSI bands: <0.1 stable,
0.1–0.25 moderate, **>0.25 significant**.

| Check | Source | Threshold | Action on fail |
|---|---|---|---|
| Data drift (text-length, rating, source mix) | Training frame vs. recent silver window | any column with PSI > `DRIFT_PSI_THRESHOLD` (0.25) | Flag in dashboard; in the cycle gate, block promotion |
| Target drift (label distribution) | Training labels vs. rating-derived labels on the current window | label PSI > `DRIFT_PSI_THRESHOLD` (0.25) | Flag in dashboard; in the cycle gate, block promotion |
| Performance — macro-F1 | Model scored on reference vs. current | F1-macro drop > `DRIFT_F1_DROP_THRESHOLD` (3%) | **Block** `Staging → Production` |
| Performance — negative-class recall | Model scored on reference vs. current | recall_neg drop > `DRIFT_RECALL_NEG_DROP_THRESHOLD` (5%) | **Block** `Staging → Production` |

The gate (`evaluate()`) blocks when **feature drift OR label drift OR either
performance metric** regresses. The negative class is treated as business-critical
(catching negative brand sentiment matters most), so it has its own recall guard
alongside F1.
`drift_score` (Evidently's share of drifted columns) is still recorded and shown
on the dashboard gauge, but it no longer drives the block decision.

## How the gate works (`evaluate`)

1. `compute_drift(reference, current)` — Evidently `DataDriftPreset` (+
   `TargetDriftPreset` when a `label` column is present), both using the PSI
   stattest at `DRIFT_PSI_THRESHOLD`, restricted to the monitored columns shared
   by both frames. Free text is excluded. Any single feature column **or** the
   label over the PSI threshold flags drift and blocks promotion.
2. If a `model` is given, score it **once per side** to get macro-F1 and
   negative-class recall (one `predict` call each — never two, so stateful models
   compare correctly).
3. Upload the HTML report to MinIO and append the `monitoring_reports` row
   **before** any block is raised, so a failing report stays in the dashboard.
4. Return `blocked_promotion`; `promote` honours it. With `raise_on_block=True`
   the task fails red via `PromotionBlocked`.

## Drift alert (human-in-the-loop)

`evaluate_and_monitor` is a read-only observer: it ends with `should_alert` (a
`ShortCircuitOperator` reading the drift result) → `send_alert`. On a blocked gate
it writes the `monitoring_reports` row and surfaces the alert in the Airflow logs.
There is **no** automatic retrain — a human reviews the dashboard and, if
warranted, retrains off-cycle by triggering `medallion_pipeline` with
`FORCE_TRAIN=1`.

## Outputs

- HTML report uploaded to MinIO under `s3://monitoring/{YYYY-MM-DD}/<type>.html`
  (`data_drift` for the observational DAG, `performance` for the cycle gate)
- One row appended to Postgres `monitoring_reports` (run_date, report_type,
  report_url, drift_score, blocked_promotion)
- The dashboard reads the latest row and embeds the HTML

## Config (env)

| Var | Default | Meaning |
|---|---|---|
| `DRIFT_PSI_THRESHOLD` | `0.25` | Per-column PSI above which a column counts as drifted (data + label) |
| `DRIFT_THRESHOLD` | `0.3` | Legacy share-of-columns value; recorded + shown on the dashboard gauge, no longer gates |
| `DRIFT_F1_DROP_THRESHOLD` | `0.03` | Macro-F1 drop that blocks |
| `DRIFT_RECALL_NEG_DROP_THRESHOLD` | `0.05` | Negative-class recall drop that blocks |
| `DRIFT_RECENT_PARTITIONS` | `7` | How many recent silver `review_date=` partitions form `current` |
| `DRIFT_SILVER_ROOT` / `DRIFT_SAMPLE_CSV` | container/repo defaults | Override input locations |

When no silver partitions exist yet, the observational check degrades to
train-vs-itself (zero drift, gate passes) so the DAG stays green while wiring settles.
