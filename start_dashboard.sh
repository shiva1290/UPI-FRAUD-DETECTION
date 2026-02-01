#!/bin/bash

echo "======================================================================"
echo " UPI FRAUD DETECTION - WEB DASHBOARD"
echo "======================================================================"
echo ""
echo "Starting the web server..."
echo ""
echo "🌐 Dashboard will be available at:"
echo "   http://localhost:5000"
echo ""
echo "Features:"
echo "  ✓ Real-time fraud detection testing"
echo "  ✓ Interactive visualizations"
echo "  ✓ Model performance comparison"
echo "  ✓ LLM predictions with reasoning"
echo "  ✓ Transaction data exploration"
echo ""
echo "======================================================================"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

cd src

# Activate virtual environment
source ../venv/bin/activate

# Start Flask app
python app.py
