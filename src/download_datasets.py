"""
Download and analyze real UPI transaction datasets from Kaggle
"""

import kagglehub
import pandas as pd
import os
import shutil

def download_datasets():
    """Download both Kaggle UPI datasets"""
    
    print("="*70)
    print("DOWNLOADING REAL UPI DATASETS FROM KAGGLE")
    print("="*70)
    
    datasets = []
    
    # Dataset 1
    print("\n📥 Downloading Dataset 1: bijitda/upi-transactions-dataset")
    try:
        path1 = kagglehub.dataset_download("bijitda/upi-transactions-dataset")
        print(f"✓ Downloaded to: {path1}")
        datasets.append(('bijitda', path1))
    except Exception as e:
        print(f"❌ Error downloading dataset 1: {e}")
        datasets.append(('bijitda', None))
    
    # Dataset 2
    print("\n📥 Downloading Dataset 2: skullagos5246/upi-transactions-2024-dataset")
    try:
        path2 = kagglehub.dataset_download("skullagos5246/upi-transactions-2024-dataset")
        print(f"✓ Downloaded to: {path2}")
        datasets.append(('skullagos5246', path2))
    except Exception as e:
        print(f"❌ Error downloading dataset 2: {e}")
        datasets.append(('skullagos5246', None))
    
    return datasets


def analyze_dataset(name, path):
    """Analyze a downloaded dataset"""
    
    if path is None:
        print(f"\n⚠️  Dataset {name} not available")
        return None
    
    print(f"\n{'='*70}")
    print(f"ANALYZING DATASET: {name}")
    print(f"{'='*70}")
    
    # List all files in the directory
    print(f"\n📁 Files in {path}:")
    for root, dirs, files in os.walk(path):
        for file in files:
            file_path = os.path.join(root, file)
            size = os.path.getsize(file_path) / (1024 * 1024)  # MB
            print(f"  - {file} ({size:.2f} MB)")
            
            # Try to load CSV files
            if file.endswith('.csv'):
                try:
                    df = pd.read_csv(file_path)
                    
                    print(f"\n📊 Dataset Info for {file}:")
                    print(f"  Shape: {df.shape}")
                    print(f"  Columns: {list(df.columns)}")
                    print(f"\n  Column Types:")
                    print(df.dtypes)
                    
                    print(f"\n  First 5 rows:")
                    print(df.head())
                    
                    print(f"\n  Statistical Summary:")
                    print(df.describe())
                    
                    print(f"\n  Missing Values:")
                    print(df.isnull().sum())
                    
                    # Check for fraud column
                    fraud_cols = [col for col in df.columns if 'fraud' in col.lower() or 'label' in col.lower()]
                    if fraud_cols:
                        print(f"\n  Fraud Distribution:")
                        for col in fraud_cols:
                            print(f"    {col}: {df[col].value_counts().to_dict()}")
                    
                    return df
                    
                except Exception as e:
                    print(f"  ❌ Error reading {file}: {e}")
    
    return None


def main():
    # Download datasets
    datasets = download_datasets()
    
    # Analyze each dataset
    dfs = []
    for name, path in datasets:
        df = analyze_dataset(name, path)
        if df is not None:
            dfs.append((name, df))
    
    print(f"\n{'='*70}")
    print("DOWNLOAD COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Successfully analyzed {len(dfs)} dataset(s)")
    
    # Save paths for later use
    with open('../data/kaggle_dataset_paths.txt', 'w') as f:
        for name, path in datasets:
            if path:
                f.write(f"{name}: {path}\n")
    
    print("\n✓ Dataset paths saved to data/kaggle_dataset_paths.txt")


if __name__ == "__main__":
    main()
