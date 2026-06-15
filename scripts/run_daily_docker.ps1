# Run the medallion pipeline inside the data-pipeline compose service.
#
# Usage (from repo root):
#   .\scripts\run_daily_docker.ps1
#   .\scripts\run_daily_docker.ps1 --run-date 2026-06-10 --sources yelp tripadvisor
#   .\scripts\run_daily_docker.ps1 --skip-bronze
#
# Requires Docker Desktop. Raw datasets are mounted from ../../Yelp_JSON and
# ../../TripAdvisor_data_cleaned.csv (override via YELP_DATA_HOST / TRIPADVISOR_DATA_HOST in .env).

$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

if (-not (Test-Path .env)) {
    Copy-Item infra/.env.example .env
    Write-Host "Created .env from infra/.env.example"
}

$runDate = (Get-Date -Format "yyyy-MM-dd")
$defaultArgs = @("--run-date", $runDate, "--sources", "yelp", "tripadvisor")
$extraArgs = if ($args.Count -gt 0) { $args } else { $defaultArgs }

docker compose --profile pipeline run --rm data-pipeline @extraArgs
