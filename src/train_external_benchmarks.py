"""
External Dataset Benchmark Training Script
Trains Random Forest models on external fraud datasets (ULB Credit Card, PaySim)
with progress logging and result saving.

This script runs separately from main training to handle large datasets efficiently.
"""

import os
import sys
import logging
import time
from datetime import datetime
from pathlib import Path

# Setup logging
log_dir = Path(__file__).parent.parent / "logs"
log_dir.mkdir(exist_ok=True)

log_file = log_dir / f"external_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from external_benchmark import run_external_benchmarks


def main():
    """Main function to run external dataset benchmarks."""
    logger.info("="*70)
    logger.info(" EXTERNAL DATASET BENCHMARK TRAINING")
    logger.info("="*70)
    logger.info(f"Log file: {log_file}")
    logger.info("")
    
    # Get source directory
    src_dir = os.path.dirname(os.path.abspath(__file__))
    logger.info(f"Source directory: {src_dir}")
    
    # Check datasets
    data_dir = os.path.join(src_dir, "..", "data")
    data_dir_abs = os.path.abspath(data_dir)
    logger.info(f"Data directory: {data_dir_abs}")
    logger.info("")
    
    # Check available datasets
    from dataset_loader import DatasetLoader
    dataset_info = DatasetLoader.get_dataset_info(data_dir_abs)
    
    logger.info("Checking for available datasets...")
    available_count = 0
    for name, info in dataset_info.items():
        status = "✓ Available" if info["available"] else "✗ Not found"
        logger.info(f"  {name}: {status}")
        if info["available"]:
            available_count += 1
    
    if available_count == 0:
        logger.warning("No external datasets found!")
        logger.info("Please download datasets from Kaggle:")
        logger.info("  - ULB Credit Card: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
        logger.info("  - PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1")
        return
    
    logger.info("")
    logger.info(f"Found {available_count} dataset(s) to benchmark")
    logger.info("Starting benchmark training...")
    logger.info("")
    
    # Run benchmarks with timing
    start_time = time.time()
    
    try:
        results = run_external_benchmarks(src_dir)
        
        elapsed_time = time.time() - start_time
        
        logger.info("")
        logger.info("="*70)
        if results:
            logger.info(f"✓ BENCHMARK TRAINING COMPLETED SUCCESSFULLY!")
            logger.info(f"  Processed {len(results)} dataset(s)")
            logger.info(f"  Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.2f} minutes)")
            logger.info("")
            logger.info("Results saved to: results/external_benchmark.csv")
            logger.info("")
            logger.info("Benchmark Summary:")
            for result in results:
                logger.info(f"  Dataset: {result['dataset']}")
                logger.info(f"    Samples: {result['samples']:,}")
                logger.info(f"    Fraud Rate: {result['fraud_rate']:.4%}")
                logger.info(f"    F1-Score: {result['f1_score']:.4f}")
                logger.info(f"    ROC-AUC: {result['roc_auc']:.4f}")
                logger.info("")
        else:
            logger.warning("No benchmarks completed (datasets may have failed to process)")
        
        logger.info("="*70)
        logger.info(f"Full log saved to: {log_file}")
        
    except KeyboardInterrupt:
        logger.warning("")
        logger.warning("Training interrupted by user (Ctrl+C)")
        logger.warning(f"Partial results may be available in results/external_benchmark.csv")
        logger.warning(f"Log file: {log_file}")
        sys.exit(1)
    except Exception as e:
        logger.error("")
        logger.error(f"ERROR during benchmark training: {e}", exc_info=True)
        logger.error(f"Log file: {log_file}")
        sys.exit(1)


if __name__ == "__main__":
    main()
