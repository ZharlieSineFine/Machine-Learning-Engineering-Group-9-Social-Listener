# scripts/demo_spike.ps1
# DEMO STEP 2 - inject a sudden negative-review spike and trigger the MLOps response.
#
#   .\scripts\demo_spike.ps1
#
# What happens (all under ~1 min):
#   1. The champion model scores today's review burst -> reviews table (negative % jumps).
#   2. Airflow's evaluate_and_monitor detects drift, blocks the gate, flags the report,
#      and triggers the medallion_train_cycle retrain DAG.
#   3. The dashboard's latest batch turns red and the spike alert fires.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)   # repo root
$py = ".\.venv\Scripts\python.exe"

# 2026-06-21 is the spike day baked into the demo data (demo_jun2026_spike.csv).
# --as-now stamps the rows as arriving *today* so they become the latest batch.
$SPIKE_DAY = "2026-06-21"
$ds = (Get-Date).ToString("yyyy-MM-dd")            # Airflow logical date for the drift task

$env:POSTGRES_HOST = "localhost"; $env:POSTGRES_PORT = "5432"
$env:POSTGRES_USER = "mlops"; $env:POSTGRES_PASSWORD = "mlops"; $env:POSTGRES_DB = "sentiment"

Write-Host "================================================================" -ForegroundColor Red
Write-Host "  NEGATIVE-REVIEW SPIKE  -  simulating a brand crisis hitting today" -ForegroundColor Red
Write-Host "================================================================" -ForegroundColor Red
Write-Host ""

Write-Host "==> [1/2] Inference: scoring today's review burst with the champion model..." -ForegroundColor Cyan
& $py -m serving.batch_infer --scenario spike --asof $SPIKE_DAY --n-recent 1 --as-now --clear-today

Write-Host ""
Write-Host "==> [2/2] Airflow: evaluate_and_monitor detecting drift (Evidently)..." -ForegroundColor Cyan
docker exec -e DRIFT_REPLAY_SCENARIO=spike -e DRIFT_REPLAY_ASOF=$SPIKE_DAY -e DRIFT_REPLAY_N_RECENT=1 `
  sentiment-airflow-scheduler airflow tasks test evaluate_and_monitor compute_and_log_drift $ds 2>&1 |
  Select-String -Pattern "replay-drift:spike|drift_score=|blocked=True|evaluate_and_monitor\]"

Write-Host ""
Write-Host "    Drift blocked the gate -> triggering retrain (medallion_train_cycle)..." -ForegroundColor Cyan
docker exec sentiment-airflow-scheduler airflow dags trigger medallion_train_cycle 2>&1 |
  Select-String -Pattern "queued|created|medallion_train_cycle"

Write-Host ""
Write-Host "DONE. The dashboard now shows the spike + the red alert banner." -ForegroundColor Green
Write-Host "  Dashboard : http://localhost:8501   (negative % jumps, alert fires)" -ForegroundColor Green
Write-Host "  Airflow   : http://localhost:8080   (medallion_train_cycle now has a fresh run)"
