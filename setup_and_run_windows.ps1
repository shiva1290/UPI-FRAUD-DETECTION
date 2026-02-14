# UPI Fraud Detection - Setup & Run (Windows PowerShell)
# Usage: .\setup_and_run_windows.ps1
# If execution is restricted, run: Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

Write-Host "======================================================================"
Write-Host " UPI FRAUD DETECTION - SETUP & RUN (Windows)"
Write-Host "======================================================================"

# Create venv if missing
if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
    Write-Host "Creating virtual environment..."
    python -m venv venv
}

& "venv\Scripts\Activate.ps1"

# Install dependencies
Write-Host "Installing dependencies..."
pip install -q -r requirements.txt

# Train models if not already present
$needTrain = -not (Test-Path "models\best_model_random_forest.pkl") -or -not (Test-Path "models\preprocessor.pkl")
if ($needTrain) {
    Write-Host ""
    Write-Host "Training models (this may take a few minutes)..."
    Set-Location src
    $hasKey = $false
    if (Test-Path "..\.env") {
        $content = Get-Content "..\.env" -Raw
        if ($content -match 'GROQ_API_KEY=.+') { $hasKey = $true }
    }
    if ($hasKey) { python train.py --with-llm } else { python train.py }
    Set-Location ..
    Write-Host ""
} else {
    Write-Host "Models found. Skip training. (Delete models\ to retrain.)"
}

# Start dashboard
Write-Host "======================================================================"
Write-Host " UPI FRAUD DETECTION - WEB DASHBOARD"
Write-Host "======================================================================"
Write-Host ""
Write-Host "Dashboard at http://localhost:5000"
Write-Host "Press Ctrl+C to stop."
Write-Host "======================================================================"
Set-Location src
python app.py
