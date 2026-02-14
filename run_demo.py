"""
Quick Demo Script - Run fraud detection pipeline and generate results
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    subprocess.check_call([
        "venv/bin/python", "-m", "pip", "install", "-q",
        "numpy", "pandas", "scikit-learn", "xgboost", 
        "matplotlib", "seaborn", "imbalanced-learn", "joblib"
    ])
    print("✓ Packages installed!\n")

def run_pipeline():
    """Run the complete training pipeline"""
    os.chdir('src')
    subprocess.call(["../venv/bin/python", "train.py", "--with-llm"])

if __name__ == "__main__":
    install_requirements()
    run_pipeline()
    
    print("\n" + "="*70)
    print(" ALL DONE! Check the results/ folder for outputs")
    print("="*70)
