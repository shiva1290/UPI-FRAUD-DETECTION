"""
External dataset benchmarks (SRP).
Evaluates the Random Forest model on external fraud datasets
such as the ULB Credit Card dataset (Kaggle) or PaySim, and reports metrics.
"""

import os
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from models import FraudDetectionModel
from dataset_loader import DatasetLoader


def _benchmark_credit_card(base_dir: str) -> Dict[str, Any]:
    """
    Benchmark on the ULB Credit Card Fraud Detection dataset from Kaggle, if available.
    
    Dataset: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    Expected file: ../data/creditcard.csv relative to src/
    
    Dataset structure:
    - Features: V1-V28 (PCA-transformed features), Time, Amount
    - Target: Class (0=normal, 1=fraud)
    - Highly imbalanced: ~0.17% fraud rate
    
    Args:
        base_dir: Base directory (typically src/)
        
    Returns:
        Dict with benchmark metrics, or empty dict if dataset not found
    """
    data_dir = os.path.join(base_dir, "..", "data")
    
    # Use DatasetLoader for consistent loading and validation
    result = DatasetLoader.load_ulb_credit_card(data_dir)
    if result is None:
        return {}
    
    df, target_col = result
    
    print("\n[External Benchmark] ULB Credit Card dataset (Kaggle) detected.")
    print(f"  Dataset URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    # Handle highly imbalanced dataset with stratified split
    # Use smaller test size for imbalanced data to ensure fraud samples in test set
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
    except ValueError:
        # Fallback if stratification fails (e.g., too few fraud samples)
        print("  ⚠️  Stratified split failed, using random split")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
    
    print(f"  Training: {len(X_train)} samples ({y_train.sum()} fraud, {y_train.mean():.4%})")
    print(f"  Testing: {len(X_test)} samples ({y_test.sum()} fraud, {y_test.mean():.4%})")
    
    # For very large datasets, use a sample for faster training
    max_training_samples = 500000  # Limit training to 500K samples for performance
    if len(X_train) > max_training_samples:
        print(f"  ⚠️  Large dataset detected ({len(X_train):,} samples)")
        print(f"  Sampling {max_training_samples:,} samples for training to improve performance...")
        # Stratified sampling to maintain fraud rate
        from sklearn.model_selection import train_test_split as tts
        X_train_sample, _, y_train_sample, _ = tts(
            X_train, y_train,
            train_size=max_training_samples,
            random_state=42,
            stratify=y_train
        )
        X_train = X_train_sample
        y_train = y_train_sample
        print(f"  Using {len(X_train):,} samples for training")
    
    print("  Training Random Forest model (this may take several minutes for large datasets)...")
    import time
    train_start = time.time()
    
    # Train Random Forest model
    model = FraudDetectionModel(model_type="random_forest", random_state=42)
    model.train(X_train, y_train, use_smote=False)
    
    train_time = time.time() - train_start
    print(f"  ✓ Model trained in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")
    
    # Evaluate on test set
    y_scores = model.predict_proba(X_test)
    y_pred = (y_scores >= 0.5).astype(int)
    
    metrics = {
        "dataset": "ULB_Credit_Card",
        "samples": int(len(df)),
        "fraud_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_scores)) if len(np.unique(y_test)) > 1 else 0.0,
    }
    
    print(
        f"  ✓ ULB Credit Card Results - "
        f"F1={metrics['f1_score']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}"
    )
    return metrics


def _benchmark_paysim(base_dir: str) -> Dict[str, Any]:
    """
    Benchmark on a PaySim-style dataset, if available.
    
    Expected file: ../data/paysim.csv with target column 'isFraud'.
    
    Args:
        base_dir: Base directory (typically src/)
        
    Returns:
        Dict with benchmark metrics, or empty dict if dataset not found
    """
    data_dir = os.path.join(base_dir, "..", "data")
    
    # Use DatasetLoader for consistent loading and validation
    result = DatasetLoader.load_paysim(data_dir)
    if result is None:
        return {}
    
    df, target_col = result
    
    print("\n[External Benchmark] PaySim dataset detected.")
    
    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    # Handle categorical columns (PaySim has 'type' column with values like 'CASH_IN', 'PAYMENT', etc.)
    categorical_cols = X.select_dtypes(include=['object']).columns
    if len(categorical_cols) > 0:
        print(f"  Encoding {len(categorical_cols)} categorical columns: {list(categorical_cols)}")
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
    
    # Convert all columns to numeric (handle any remaining non-numeric)
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop any columns that couldn't be converted
    X = X.select_dtypes(include=[np.number])
    
    # Handle any remaining NaN values
    X = X.fillna(0)
    
    # Remove any infinite values
    X = X.replace([np.inf, -np.inf], 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    print(f"  Training: {len(X_train)} samples ({y_train.sum()} fraud, {y_train.mean():.4%})")
    print(f"  Testing: {len(X_test)} samples ({y_test.sum()} fraud, {y_test.mean():.4%})")
    
    # For very large datasets, use a sample for faster training
    max_training_samples = 500000  # Limit training to 500K samples for performance
    if len(X_train) > max_training_samples:
        print(f"  ⚠️  Large dataset detected ({len(X_train):,} samples)")
        print(f"  Sampling {max_training_samples:,} samples for training to improve performance...")
        # Stratified sampling to maintain fraud rate
        from sklearn.model_selection import train_test_split as tts
        X_train_sample, _, y_train_sample, _ = tts(
            X_train, y_train,
            train_size=max_training_samples,
            random_state=42,
            stratify=y_train
        )
        X_train = X_train_sample
        y_train = y_train_sample
        print(f"  Using {len(X_train):,} samples for training")
    
    print("  Training Random Forest model (this may take several minutes for large datasets)...")
    import time
    train_start = time.time()
    
    model = FraudDetectionModel(model_type="random_forest", random_state=42)
    model.train(X_train, y_train, use_smote=False)
    
    train_time = time.time() - train_start
    print(f"  ✓ Model trained in {train_time:.2f} seconds ({train_time/60:.2f} minutes)")

    y_scores = model.predict_proba(X_test)
    y_pred = (y_scores >= 0.5).astype(int)

    metrics = {
        "dataset": "PaySim",
        "samples": int(len(df)),
        "fraud_rate": float(y.mean()),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_scores)) if len(np.unique(y_test)) > 1 else 0.0,
    }

    print(
        f"  ✓ PaySim Results - "
        f"F1={metrics['f1_score']:.4f}, "
        f"Precision={metrics['precision']:.4f}, "
        f"Recall={metrics['recall']:.4f}, "
        f"ROC-AUC={metrics['roc_auc']:.4f}"
    )
    return metrics


def run_external_benchmarks(src_dir: str) -> List[Dict[str, Any]]:
    """
    Run all available external benchmarks and save results to results/external_benchmark.csv.
    
    Checks for:
    - ULB Credit Card Fraud Detection (Kaggle): https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
    - PaySim synthetic financial dataset
    
    Args:
        src_dir: Source directory (typically src/)
        
    Returns:
        A list of per-dataset metrics dictionaries.
    """
    results: List[Dict[str, Any]] = []
    # Normalize data directory path (resolve relative paths)
    data_dir = os.path.normpath(os.path.join(src_dir, "..", "data"))
    data_dir_abs = os.path.abspath(data_dir)
    
    # Check dataset availability
    dataset_info = DatasetLoader.get_dataset_info(data_dir_abs)
    
    print("\n[External Benchmark] Checking for available datasets...")
    print(f"  Data directory: {data_dir_abs}")
    for name, info in dataset_info.items():
        status = "✓ Available" if info["available"] else "✗ Not found"
        print(f"  {name}: {status}")
        if not info["available"]:
            print(f"    Download from: {info['url']}")
            print(f"    Place file at: {info['path']}")
    
    # Run benchmarks for available datasets
    credit_metrics = _benchmark_credit_card(src_dir)
    if credit_metrics:
        results.append(credit_metrics)

    paysim_metrics = _benchmark_paysim(src_dir)
    if paysim_metrics:
        results.append(paysim_metrics)

    if results:
        df = pd.DataFrame(results)
        out_path = os.path.join(src_dir, "..", "results", "external_benchmark.csv")
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"\n✓ External benchmark results saved to {out_path}")

        # Short cross-dataset discussion for console / report
        print("\n[Cross-Dataset Performance Discussion]")
        print("-" * 70)
        for row in results:
            print(
                f"  Dataset: {row['dataset']:<20} | "
                f"Fraud Rate: {row['fraud_rate']:>7.3%} | "
                f"F1: {row['f1_score']:.4f} | "
                f"ROC-AUC: {row['roc_auc']:.4f}"
            )
        print(
            "\n  Note: Differences across datasets reflect varying fraud rates, feature spaces, "
            "and noise levels. The same Random Forest may achieve higher ROC-AUC on one "
            "dataset while having lower precision/recall on another; this is expected "
            "and underscores the need to tune thresholds and costs per deployment context."
        )
    else:
        print("\n⚠️  No external datasets found. External benchmarks are optional.")
        print("  To add benchmarks:")
        print("  1. Download datasets from Kaggle (requires free account):")
        print("     - ULB Credit Card: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        print("     - PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1")
        print(f"  2. Place CSV files in: {os.path.normpath(os.path.abspath(data_dir))}")
        print("  3. Re-run training to generate benchmark results")
        print("  Note: External benchmarks are optional and do not affect main model training.")

    return results

