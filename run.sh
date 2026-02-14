#!/bin/bash

echo "======================================================================"
echo " UPI FRAUD DETECTION - QUICK START SCRIPT"
echo "======================================================================"
echo ""
echo "This will run the complete fraud detection pipeline and generate:"
echo "  ✓ 50,000 synthetic UPI transactions"
echo "  ✓ Train 5 different ML models"
echo "  ✓ Generate performance metrics and visualizations"
echo "  ✓ Save results for your research paper"
echo ""
echo "Estimated time: 2-3 minutes"
echo "======================================================================"
echo ""

cd src

# Activate virtual environment
source ../venv/bin/activate

# Load environment variables from .env file (parent directory)
if [ -f ../.env ]; then
    # Handle Windows line endings and comments safely
    export $(grep -v '^#' ../.env | sed 's/\r$//' | xargs)
fi

# Run the training pipeline
# python train.py
python train.py --with-llm

echo ""
echo "======================================================================"
echo " RESULTS LOCATION"
echo "======================================================================"
echo "  📊 Performance metrics: ../results/model_performance.csv"
echo "  📈 Comparison chart: ../results/model_comparison.png"
echo "  📉 Confusion matrix: ../results/confusion_matrix_*.png"
echo "  📊 ROC curve: ../results/roc_curve_*.png"
echo "  💾 Best model: ../models/best_model_*.pkl"
echo "======================================================================"
echo ""
echo "✅ You can now use these results in your research paper!"
echo ""
