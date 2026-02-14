@echo off
REM UPI Fraud Detection - Setup & Run (Windows Command Prompt)
REM Usage: double-click or run: setup_and_run_windows.bat

cd /d "%~dp0"

echo ======================================================================
echo  UPI FRAUD DETECTION - SETUP ^& RUN (Windows)
echo ======================================================================

REM Create venv if missing
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install -q -r requirements.txt

REM Train models if not already present
if not exist "models\best_model_random_forest.pkl" (
    echo.
    echo Training models (this may take a few minutes)...
    cd src
    findstr /R "GROQ_API_KEY=." ..\\.env >nul 2>&1 && python train.py --with-llm || python train.py
    cd ..
    echo.
) else if not exist "models\preprocessor.pkl" (
    echo.
    echo Training models (preprocessor missing)...
    cd src
    findstr /R "GROQ_API_KEY=." ..\\.env >nul 2>&1 && python train.py --with-llm || python train.py
    cd ..
    echo.
) else (
    echo Models found. Skip training. Delete models\ to retrain.
)

REM Start dashboard
echo ======================================================================
echo  UPI FRAUD DETECTION - WEB DASHBOARD
echo ======================================================================
echo.
echo Dashboard at http://localhost:5000
echo Press Ctrl+C to stop.
echo ======================================================================
cd src
python app.py
pause
