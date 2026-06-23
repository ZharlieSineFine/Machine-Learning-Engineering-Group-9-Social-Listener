# scripts/demo_up.ps1
# DEMO STEP 1 - bring the full stack up (incl MLflow registry + FastAPI serving) and
# seed a clean (normal) day on the dashboard.
#
#   .\scripts\demo_up.ps1
#
# Result: http://localhost:8501 shows a normal day; the API serves the champion from
# MLflow; the MLOps Monitor page shows the registry + Production/Staging shadow panel.
$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)   # repo root
$py = ".\.venv\Scripts\python.exe"

function Wait-Url($url, $tries = 40) {
    for ($i = 0; $i -lt $tries; $i++) {
        try { if ((Invoke-WebRequest -UseBasicParsing $url -TimeoutSec 3).StatusCode -eq 200) { return $true } } catch {}
        Start-Sleep 2
    }
    return $false
}

Write-Host "==> [1/6] Starting services (postgres, minio, mlflow, airflow, dashboard)..." -ForegroundColor Cyan
docker compose up -d postgres minio minio-init mlflow airflow-init airflow-webserver airflow-scheduler dashboard | Out-Null

Write-Host "==> [2/6] Waiting for Postgres, MLflow, Airflow..." -ForegroundColor Cyan
foreach ($i in 1..40) { docker exec sentiment-postgres pg_isready -U mlops *>$null; if ($LASTEXITCODE -eq 0) { break }; Start-Sleep 2 }
[void](Wait-Url "http://localhost:5001/")
foreach ($i in 1..40) { docker exec sentiment-airflow-scheduler airflow version *>$null; if ($LASTEXITCODE -eq 0) { break }; Start-Sleep 2 }

Write-Host "==> [3/6] Registering the champion in MLflow (idempotent)..." -ForegroundColor Cyan
docker exec sentiment-airflow-scheduler python /opt/project/scripts/register_champion.py 2>&1 | Select-String -Pattern "register\]"

Write-Host "==> [4/6] Starting the API (serves sentiment-baseline/Production from MLflow)..." -ForegroundColor Cyan
docker compose up -d api | Out-Null

# Host-side Postgres access for the inference below. IMPORTANT: set this AFTER
# 'docker compose up' so it cannot leak into the containers' ${POSTGRES_HOST}.
$env:POSTGRES_HOST = "localhost"; $env:POSTGRES_PORT = "5432"
$env:POSTGRES_USER = "mlops"; $env:POSTGRES_PASSWORD = "mlops"; $env:POSTGRES_DB = "sentiment"

Write-Host "==> [5/6] Generating replay + scoring a clean 2-week history -> reviews table..." -ForegroundColor Cyan
& $py -m data.ingest.replay --scenario stable | Out-Null
& $py -m data.ingest.replay --scenario spike  | Out-Null
& $py -m serving.batch_infer --scenario stable --shift-to-today --truncate
# Reset the drift signal on a fresh bring-up. We REPLACE the rows rather than just
# TRUNCATE-ing: the dashboard reads monitoring_reports, and when the table is EMPTY
# its panel falls back to a live `import monitoring` check the dashboard container
# can't satisfy (no monitoring mount / PYTHONPATH) -> "No module named monitoring".
# We seed ~2 weeks of low-PSI rows so the drift sparkline has a steady baseline for
# the spike (demo_spike) to stand against, then a clean 0 for today. Units are label
# PSI (what evaluate_and_monitor now records) — NOT the old share-of-drifted-columns.
$resetSql = @"
TRUNCATE monitoring_reports;
INSERT INTO monitoring_reports (run_date, report_type, report_url, drift_score, blocked_promotion)
SELECT d::date, 'data_drift', 'baseline backfill (demo seed)',
       round((0.003 + random()*0.025)::numeric, 4), false
FROM generate_series(CURRENT_DATE - INTERVAL '13 days',
                     CURRENT_DATE - INTERVAL '1 day', INTERVAL '1 day') AS d;
INSERT INTO monitoring_reports (run_date, report_type, report_url, drift_score, blocked_promotion)
VALUES (CURRENT_DATE, 'data_drift', 'baseline clean day (demo_up reset)', 0, false);
"@
docker exec sentiment-postgres psql -U mlops -d sentiment -c $resetSql | Out-Null

Write-Host "==> [6/6] Seeding the shadow-deploy log (API /predict/batch)..." -ForegroundColor Cyan
if (Wait-Url "http://localhost:8000/health") { & $py scripts\seed_shadow.py } else { Write-Host "    (API not ready yet; skipping shadow seed)" -ForegroundColor Yellow }

Write-Host ""
Write-Host "Stack is up and the dashboard shows a NORMAL day." -ForegroundColor Green
Write-Host "  Dashboard : http://localhost:8501   (Marketing + MLOps Monitor pages)" -ForegroundColor Green
Write-Host "  API       : http://localhost:8000/docs   (/health, /predict, /shadow/log)"
Write-Host "  MLflow    : http://localhost:5001   (sentiment-baseline: Production + Staging)"
Write-Host "  Airflow   : http://localhost:8080   (airflow / airflow)"
Write-Host ""
Write-Host "Next:  .\scripts\demo_spike.ps1   to inject the negative-review spike." -ForegroundColor Yellow
