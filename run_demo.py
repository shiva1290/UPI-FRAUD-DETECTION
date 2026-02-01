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
        sys.executable, "-m", "pip", "install", "-q",
        "numpy", "pandas", "scikit-learn", "xgboost", 
        "matplotlib", "seaborn", "imbalanced-learn", "joblib"
    ])
    print("✓ Packages installed!\n")

def run_pipeline():
    """Run the complete training pipeline"""
    os.chdir('src')
    subprocess.call([sys.executable, "train.py"])

if __name__ == "__main__":
    print("="*70)
    print(" UPI FRAUD DETECTION - QUICK START")
    print("="*70)
    print("\nThis will:")
    print("  1. Install required Python packages")
    print("  2. Generate synthetic UPI transaction data")
    print("  3. Train multiple ML models")
    print("  4. Generate performance metrics and visualizations")
    print("  5. Save results for your research paper")
    print("\nEstimated time: 3-5 minutes")
    print("="*70)
    
    input("\nPress Enter to start...")
    
    install_requirements()
    run_pipeline()
    
    print("\n" + "="*70)
    print(" ALL DONE! Check the results/ folder for outputs")
    print("="*70)
