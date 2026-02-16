#!/bin/bash
# Script to run external dataset benchmarks with logging
# This script runs separately from main training to handle large datasets

set -e

echo "=========================================="
echo "External Dataset Benchmark Training"
echo "=========================================="

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Run the benchmark training script
cd src
python train_external_benchmarks.py

echo ""
echo "=========================================="
echo "Benchmark training completed!"
echo "Check logs/ directory for detailed logs"
echo "Results saved to results/external_benchmark.csv"
echo "=========================================="
