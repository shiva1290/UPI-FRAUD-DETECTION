"""
Dataset loader utilities (SRP).
Handles loading external fraud datasets (e.g., Kaggle ULB Credit Card).
Single responsibility: dataset loading and validation.
"""

import os
import pandas as pd
from typing import Optional, Tuple
import warnings


class DatasetLoader:
    """
    Loads and validates external fraud datasets.
    Single responsibility: dataset I/O and basic validation.
    """

    @staticmethod
    def load_ulb_credit_card(data_dir: str) -> Optional[Tuple[pd.DataFrame, str]]:
        """
        Load ULB Credit Card Fraud Detection dataset from Kaggle.
        
        Dataset URL: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
        
        Expected file: data/creditcard.csv
        Expected columns: V1-V28 (PCA features), Time, Amount, Class (target: 0=normal, 1=fraud)
        
        Args:
            data_dir: Directory containing the dataset file
            
        Returns:
            Tuple of (DataFrame, target_column_name) if found, None otherwise
        """
        path = os.path.join(data_dir, "creditcard.csv")
        
        if not os.path.exists(path):
            return None
        
        try:
            print(f"[DatasetLoader] Loading ULB Credit Card dataset from {path}...")
            df = pd.read_csv(path)
            
            # Validate required columns
            required_cols = ["Class"]
            missing_cols = [col for col in required_cols if col not in df.columns]
            if missing_cols:
                print(f"⚠️  Missing required columns: {missing_cols}")
                return None
            
            # Check for expected feature columns (V1-V28 are PCA features)
            v_cols = [f"V{i}" for i in range(1, 29)]
            has_v_features = any(col in df.columns for col in v_cols)
            
            if not has_v_features:
                print("⚠️  Warning: Expected PCA features V1-V28 not found. Dataset may be in different format.")
            
            # Validate target column values
            if "Class" in df.columns:
                unique_values = df["Class"].unique()
                if not set(unique_values).issubset({0, 1}):
                    print(f"⚠️  Warning: 'Class' column contains unexpected values: {unique_values}")
                else:
                    fraud_count = df["Class"].sum()
                    fraud_rate = fraud_count / len(df)
                    print(f"✓ Dataset loaded: {len(df)} samples, {fraud_count} fraud ({fraud_rate:.2%})")
            
            return df, "Class"
            
        except Exception as e:
            print(f"⚠️  Error loading ULB Credit Card dataset: {e}")
            return None

    @staticmethod
    def load_paysim(data_dir: str) -> Optional[Tuple[pd.DataFrame, str]]:
        """
        Load PaySim dataset.
        
        Expected files: data/paysim.csv or data/PS_*.csv (PaySim format)
        Expected target column: isFraud
        
        Args:
            data_dir: Directory containing the dataset file
            
        Returns:
            Tuple of (DataFrame, target_column_name) if found, None otherwise
        """
        # Try paysim.csv first, then look for PS_*.csv files
        path = os.path.join(data_dir, "paysim.csv")
        
        if not os.path.exists(path):
            # Look for PaySim format files (PS_*.csv)
            import glob
            ps_files = glob.glob(os.path.join(data_dir, "PS_*.csv"))
            if ps_files:
                path = ps_files[0]  # Use first matching file
            else:
                return None
        
        try:
            print(f"[DatasetLoader] Loading PaySim dataset from {path}...")
            df = pd.read_csv(path)
            
            target_col = "isFraud"
            if target_col not in df.columns:
                print(f"⚠️  Missing target column '{target_col}'")
                return None
            
            fraud_count = df[target_col].sum()
            fraud_rate = fraud_count / len(df)
            print(f"✓ Dataset loaded: {len(df)} samples, {fraud_count} fraud ({fraud_rate:.2%})")
            
            return df, target_col
            
        except Exception as e:
            print(f"⚠️  Error loading PaySim dataset: {e}")
            return None

    @staticmethod
    def get_dataset_info(data_dir: str) -> dict:
        """
        Get information about available datasets.
        
        Returns:
            Dict with dataset names as keys and availability status as values
        """
        info = {}
        
        # Normalize data_dir path (resolve relative paths and symlinks)
        data_dir_abs = os.path.abspath(os.path.expanduser(data_dir))
        
        # Check ULB Credit Card
        ulb_path = os.path.join(data_dir_abs, "creditcard.csv")
        ulb_path_normalized = os.path.normpath(ulb_path)
        info["ULB_Credit_Card"] = {
            "available": os.path.exists(ulb_path_normalized),
            "path": ulb_path_normalized,
            "url": "https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud",
            "description": "ULB Credit Card Fraud Detection (Kaggle) - 284,807 transactions with 28 PCA features"
        }
        
        # Check PaySim (try paysim.csv first, then PS_*.csv files)
        paysim_path = os.path.join(data_dir_abs, "paysim.csv")
        if not os.path.exists(paysim_path):
            import glob
            ps_files = glob.glob(os.path.join(data_dir_abs, "PS_*.csv"))
            if ps_files:
                paysim_path = ps_files[0]
        paysim_path_normalized = os.path.normpath(paysim_path)
        info["PaySim"] = {
            "available": os.path.exists(paysim_path_normalized),
            "path": paysim_path_normalized,
            "url": "https://www.kaggle.com/datasets/ealaxi/paysim1",
            "description": "PaySim synthetic financial dataset"
        }
        
        return info
