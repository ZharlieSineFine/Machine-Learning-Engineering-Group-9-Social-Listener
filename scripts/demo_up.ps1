# scripts/demo_up.ps1
# DEMO STEP 1 - bring the stack up and seed a clean (normal) day on the dashboard.
#
#   docker compose up -d        # (first time / after a reboot - builds images)
#   .\scripts\demo_up.ps1       # start services + seed a normal 2-week history
#
# Result: http://localhost:8501 shows a normal day (sentiment under threshold).
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)   # repo root
$py = ".\.venv\Scripts\python.exe"

Write-Host "==> [1/4] Starting the stack (postgres, minio, mlflow, airflow, dashboard)..." -ForegroundColor Cyan
docker compose up -d postgres minio minio-init mlflow airflow-init airflow-webserver airflow-scheduler dashboard | Out-Null

Write-Host "==> [2/4] Waiting for Postgres to be ready..." -ForegroundColor Cyan
foreach ($i in 1..40) { docker exec sentiment-postgres pg_isready -U mlops *>$null; if ($LASTEXITCODE -eq 0) { break }; Start-Sleep 2 }

# Host-side Postgres access for the inference below. IMPORTANT: set this AFTER
# 'docker compose up' so it cannot leak into the containers' ${POSTGRES_HOST}.
$env:POSTGRES_HOST = "localhost"; $env:POSTGRES_PORT = "5432"
$env:POSTGRES_USER = "mlops"; $env:POSTGRES_PASSWORD = "mlops"; $env:POSTGRES_DB = "sentiment"

Write-Host "==> [3/4] Generating replay streams (clean + spike windows)..." -ForegroundColor Cyan
& $py -m data.ingest.replay --scenario stable | Out-Null
& $py -m data.ingest.replay --scenario spike  | Out-Null

Write-Host "==> [4/4] Scoring a clean 2-week history with the champion model -> reviews table..." -ForegroundColor Cyan
& $py -m serving.batch_infer --scenario stable --shift-to-today --truncate

Write-Host ""
Write-Host "Stack is up and the dashboard shows a NORMAL day." -ForegroundColor Green
Write-Host "  Dashboard : http://localhost:8501   (Marketing view -> 'Sentiment normal')" -ForegroundColor Green
Write-Host "  Airflow   : http://localhost:8080   (airflow / airflow)"
Write-Host "  MLflow    : http://localhost:5001"
Write-Host ""
Write-Host "Next:  .\scripts\demo_spike.ps1   to inject the negative-review spike." -ForegroundColor Yellow
