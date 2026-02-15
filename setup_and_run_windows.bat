@echo off
REM UPI Fraud Detection - Setup & Run (Windows)
REM Usage: bin\setup_and_run_windows.bat

cd /d "%~dp0\.."

echo ======================================================================
echo  UPI FRAUD DETECTION - SETUP ^& RUN (Windows)
echo ======================================================================

if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installing dependencies...
pip install -q -r requirements.txt

if not exist "models\best_model_random_forest.pkl" (
    echo.
    echo Training models...
    cd src
    findstr /R "GROQ_API_KEY=." ..\.env >nul 2>&1 && python train.py --with-llm || python train.py
    cd ..
) else if not exist "models\preprocessor.pkl" (
    echo.
    echo Training models (preprocessor missing)...
    cd src
    findstr /R "GROQ_API_KEY=." ..\.env >nul 2>&1 && python train.py --with-llm || python train.py
    cd ..
) else (
    echo Models found. Skip training.
)

echo ======================================================================
echo  UPI FRAUD DETECTION - WEB DASHBOARD
echo ======================================================================
echo Dashboard at http://localhost:5000
echo ======================================================================
cd src
python app.py
pause
