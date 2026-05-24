#Requires -Version 5.1
$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot ".venv\Scripts\Activate.ps1")
$env:PYTHONPATH = $PSScriptRoot
python pipeline/prefect_pipeline.py
