#!/bin/bash
# One-time setup and run: install deps, train models, start dashboard.
# For LLM features: add GROQ_API_KEY to .env and run train with --with-llm first.

set -e
cd "$(dirname "$0")"

echo "======================================================================"
echo " UPI FRAUD DETECTION - SETUP & RUN"
echo "======================================================================"

# Create venv if missing
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

source venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Train models (and generate data) if not already present
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

# Start dashboard
echo "Starting dashboard at http://localhost:5000"
echo "Press Ctrl+C to stop."
echo "======================================================================"
cd src && python app.py
