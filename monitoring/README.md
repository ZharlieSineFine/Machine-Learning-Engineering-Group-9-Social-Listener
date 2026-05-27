# Monitoring — Drift & Performance

**Owner:** Charlie + Ha

Evidently runs inside the `evaluate_and_monitor` Airflow DAG. There's no standalone Evidently service — it's a Python lib invoked daily.

## What we monitor

| Check | Source | Threshold | Action on fail |
|---|---|---|---|
| Data drift (text length distribution, source mix, language) | Train baseline vs. last 7d ingested reviews | drift score > 0.3 | Flag in dashboard |
| Target/prediction drift | Train label distribution vs. predicted labels | drift score > 0.3 | Flag in dashboard |
| Performance regression | Held-out F1 vs. running window | F1 drops > 3% | **Block** `Staging → Production` promotion |

## Outputs

- HTML report uploaded to MinIO under `s3://monitoring/{YYYY-MM-DD}/report.html`
- One row appended to Postgres `monitoring_reports` with the path, drift score, F1, gate decision
- The dashboard reads the latest row and embeds the HTML

## Phase 1 stub
Run Evidently with `DataDriftPreset` on train vs. itself (will always pass) so the wiring exists. Real reference dataset comes in phase 2.
