# UPI Fraud Detection - Setup & Run (Windows PowerShell)
# Usage: .\setup_and_run_windows.ps1  (run from project root)

$ErrorActionPreference = "Stop"
$ROOT = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $ROOT

Write-Host "======================================================================"
Write-Host " UPI FRAUD DETECTION - SETUP & RUN (Windows)"
Write-Host "======================================================================"

if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

& "venv\Scripts\Activate.ps1"

Write-Host "Installing dependencies..."
pip install -q -r requirements.txt

$needTrain = -not (Test-Path "models\best_model_random_forest.pkl") -or -not (Test-Path "models\preprocessor.pkl")
if ($needTrain) {
    Write-Host ""
    Write-Host "Training models..."
    Set-Location src
    $hasKey = $false
    if (Test-Path "..\.env") {
        $content = Get-Content "..\.env" -Raw
        if ($content -match 'GROQ_API_KEY=.+') { $hasKey = $true }
    }
    if ($hasKey) { python train.py --with-llm } else { python train.py }
    Set-Location ..
} else {
    Write-Host "Models found. Skip training."
}

Write-Host "======================================================================"
Write-Host " UPI FRAUD DETECTION - WEB DASHBOARD"
Write-Host "======================================================================"
Write-Host "Dashboard at http://localhost:5000"
Set-Location src
python app.py
