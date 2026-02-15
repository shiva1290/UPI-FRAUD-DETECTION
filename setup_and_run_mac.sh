#!/bin/bash
# UPI Fraud Detection - Setup & Run (macOS)
# Usage: ./bin/setup_and_run_mac.sh  OR  from project root: bin/setup_and_run_mac.sh

set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "======================================================================"
echo " UPI FRAUD DETECTION - SETUP & RUN (macOS)"
echo "======================================================================"

if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installing dependencies..."
pip install -q -r requirements.txt

if [ ! -f models/best_model_random_forest.pkl ] || [ ! -f models/preprocessor.pkl ]; then
    echo ""
    echo "Training models (this may take a few minutes)..."
    if [ -f .env ] && grep -q 'GROQ_API_KEY=.\+' .env 2>/dev/null; then
        cd src && python train.py --with-llm && cd ..
    else
        cd src && python train.py && cd ..
    fi
    echo ""
else
    echo "Models found. Skip training. (Delete models/ to retrain.)"
fi

echo "======================================================================"
echo " UPI FRAUD DETECTION - WEB DASHBOARD"
echo "======================================================================"
echo ""
echo "Dashboard at http://localhost:5000"
echo "Press Ctrl+C to stop."
echo "======================================================================"
cd src && python app.py
