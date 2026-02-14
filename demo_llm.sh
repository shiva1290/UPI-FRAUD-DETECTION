#!/bin/bash

echo "======================================================================"
echo " UPI FRAUD DETECTION - LLM DEMO"
echo "======================================================================"
echo ""
echo "This will test the LLM-based fraud detection with sample transactions"
echo ""
echo "Prerequisites:"
echo "  1. Groq API key configured in .env file"
echo "  2. Run: cp .env.example .env"
echo "  3. Add your API key to .env"
echo ""
echo "======================================================================"
echo ""

cd src

# Activate virtual environment
source ../venv/bin/activate

# Run the demo
python demo_llm.py

echo ""
