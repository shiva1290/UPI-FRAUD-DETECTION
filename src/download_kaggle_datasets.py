"""
Kaggle dataset downloader (SRP).
Downloads external fraud datasets from Kaggle for benchmarking.
Single responsibility: dataset downloading.
"""

import os
import sys
import subprocess
from pathlib import Path


def check_kaggle_installed():
    """Check if kaggle package is installed."""
    try:
        import kaggle
        return True
    except ImportError:
        return False


def install_kaggle():
    """Install kaggle package."""
    print("Installing kaggle package...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle", "--quiet"])
        return True
    except subprocess.CalledProcessError:
        print("⚠️  Failed to install kaggle package")
        return False


def check_kaggle_credentials():
    """Check if Kaggle API credentials are configured."""
    kaggle_dir = Path.home() / ".kaggle"
    kaggle_json = kaggle_dir / "kaggle.json"
    
    if kaggle_json.exists():
        return True
    
    # Check environment variables
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True
    
    return False


def download_dataset(dataset_name: str, output_dir: str):
    """
    Download a dataset from Kaggle.
    
    Args:
        dataset_name: Kaggle dataset identifier (e.g., "mlg-ulb/creditcardfraud")
        output_dir: Directory to save the dataset
    """
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
        
        api = KaggleApi()
        api.authenticate()
        
        print(f"Downloading {dataset_name}...")
        api.dataset_download_files(
            dataset_name,
            path=output_dir,
            unzip=True,
            quiet=False
        )
        print(f"✓ Successfully downloaded {dataset_name}")
        return True
        
    except Exception as e:
        print(f"⚠️  Error downloading {dataset_name}: {e}")
        return False


def main():
    """Main function to download external benchmark datasets."""
    print("="*70)
    print(" KAGGLE DATASET DOWNLOADER")
    print("="*70)
    
    # Check if kaggle is installed
    if not check_kaggle_installed():
        print("\n⚠️  Kaggle package not found.")
        response = input("Install kaggle package? (y/n): ").strip().lower()
        if response == 'y':
            if not install_kaggle():
                print("\n❌ Failed to install kaggle. Please install manually:")
                print("   pip install kaggle")
                return
        else:
            print("\nPlease install kaggle first:")
            print("   pip install kaggle")
            print("\nThen configure your Kaggle API credentials:")
            print("   1. Go to https://www.kaggle.com/account")
            print("   2. Scroll to 'API' section")
            print("   3. Click 'Create New API Token'")
            print("   4. Save kaggle.json to ~/.kaggle/kaggle.json")
            return
    
    # Check credentials
    if not check_kaggle_credentials():
        print("\n⚠️  Kaggle API credentials not found.")
        print("\nTo set up Kaggle API credentials:")
        print("1. Go to https://www.kaggle.com/account")
        print("2. Scroll to 'API' section")
        print("3. Click 'Create New API Token'")
        print("4. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("\nOr set environment variables:")
        print("   export KAGGLE_USERNAME=your_username")
        print("   export KAGGLE_KEY=your_api_key")
        return
    
    # Get project root and data directory
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    data_dir = project_root / "data"
    data_dir.mkdir(exist_ok=True)
    
    print(f"\nData directory: {data_dir}")
    
    # Datasets to download
    datasets = {
        "ULB Credit Card": "mlg-ulb/creditcardfraud",
        "PaySim": "ealaxi/paysim1"
    }
    
    print("\nDatasets to download:")
    for name, dataset_id in datasets.items():
        print(f"  - {name}: {dataset_id}")
    
    response = input("\nProceed with download? (y/n): ").strip().lower()
    if response != 'y':
        print("Download cancelled.")
        return
    
    # Download datasets
    success_count = 0
    for name, dataset_id in datasets.items():
        print(f"\n[{name}]")
        if download_dataset(dataset_id, str(data_dir)):
            success_count += 1
    
    print("\n" + "="*70)
    if success_count == len(datasets):
        print("✓ All datasets downloaded successfully!")
        print(f"  Files saved to: {data_dir}")
    elif success_count > 0:
        print(f"⚠️  {success_count}/{len(datasets)} datasets downloaded successfully")
    else:
        print("❌ Failed to download datasets")
    print("="*70)


if __name__ == "__main__":
    main()
