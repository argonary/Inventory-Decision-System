#Requires -Version 5.1
param(
    [string]$Version = "v_$(Get-Date -Format 'yyyy_MM_dd')",
    [double[]]$Quantiles = @(0.90, 0.95)
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root
function Invoke-Step([string]$Label, [scriptblock]$Block) {
    Write-Host ""
    Write-Host "--- $Label ---" -ForegroundColor Cyan
    & $Block
    if ($LASTEXITCODE -ne 0) {
        Write-Host ""
        Write-Host "FAILED: $Label (exit code $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
    Write-Host "OK" -ForegroundColor Green
}
$activate = Join-Path $root ".venv\Scripts\Activate.ps1"
if (-not (Test-Path $activate)) {
    Write-Host "ERROR: .venv not found. Create it with:" -ForegroundColor Red
    Write-Host "  python -m venv .venv" -ForegroundColor Yellow
    Write-Host "  pip install -r requirements.txt" -ForegroundColor Yellow
    exit 1
}
. $activate
$env:PYTHONPATH = $root
$required = @("train.csv", "items.csv", "stores.csv", "holidays_events.csv", "oil.csv")
$missing = $required | Where-Object { -not (Test-Path "data\raw\$_") }
if ($missing) {
    Write-Host "ERROR: Missing raw data files in data/raw/:" -ForegroundColor Red
    $missing | ForEach-Object { Write-Host "  $_" -ForegroundColor Red }
    Write-Host ""
    Write-Host "Download the Corporacion Favorita dataset from Kaggle and place the CSVs in data/raw/." -ForegroundColor Yellow
    exit 1
}
Write-Host "Rebuild pipeline starting - version: $Version  quantiles: $($Quantiles -join ', ')" -ForegroundColor Green
Invoke-Step "Build base training snapshot 2013-2015" {
    python scripts/build_training_snapshot.py
}
Invoke-Step "Build featured training snapshot" {
    python scripts/build_featured_snapshot.py
}
Invoke-Step "Build base test snapshot 2016 Q1" {
    python scripts/build_test_snapshot_2016Q1.py
}
Invoke-Step "Build featured test snapshot 2016 Q1" {
    python scripts/build_test_featured_snapshot_2016Q1.py
}
Invoke-Step "Train quantile models and update latest" {
    python scripts/train_quantile_model.py --version $Version --quantiles $Quantiles --update-latest
}
Write-Host ""
Write-Host "Pipeline complete. Version $Version is now active." -ForegroundColor Green
Write-Host "Launch the app with: .\run_app.ps1" -ForegroundColor Cyan

