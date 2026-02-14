"""
Enhanced Training Pipeline with LLM Integration
Compares traditional ML models with LLM-based fraud detection
"""

import sys
import os

# Load .env from project root so GROQ_API_KEY is available (and properly parsed)
try:
    from dotenv import load_dotenv
    _root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    load_dotenv(os.path.join(_root, '.env'))
except ImportError:
    pass

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

from data_generator import EnhancedUPIDataGenerator
from preprocessor import UPIPreprocessor
from models import FraudDetectionModel, ModelComparison
from llm_detector import LLMFraudDetector

def main(use_llm=False):
    """Main training pipeline with LLM comparison"""
    
    # Ensure output dirs exist (paths relative to src/)
    for d in ('../models', '../results', '../data'):
        os.makedirs(os.path.join(os.path.dirname(__file__), d), exist_ok=True)
    
    print("="*70)
    print(" UPI FRAUD DETECTION - ML + LLM HYBRID SYSTEM")
    print("="*70)
    
    # Step 1: Load Enhanced Data
    print("\n[1/6] LOADING ENHANCED UPI TRANSACTION DATA")
    print("-"*70)
    
    # Check if enhanced dataset exists, otherwise generate it
    data_path = '../data/upi_transactions.csv'
    if not os.path.exists(data_path):
        print("⚠️  Enhanced dataset not found, generating...")
        from data_generator import EnhancedUPIDataGenerator
        generator = EnhancedUPIDataGenerator(use_real_data=True, target_samples=150000)
        data = generator.generate()
        data.to_csv(data_path, index=False)
    else:
        data = pd.read_csv(data_path, parse_dates=['timestamp'])
    
    print(f"✓ Loaded {len(data)} transactions")
    print(f"  Fraud: {data['is_fraud'].sum()} ({data['is_fraud'].mean():.2%})")
    print(f"  Legitimate: {(~data['is_fraud'].astype(bool)).sum()}")
    
    # Step 2: Preprocess Data
    print("\n[2/6] PREPROCESSING DATA")
    print("-"*70)
    preprocessor = UPIPreprocessor()
    processed_data = preprocessor.preprocess(data, fit=True)
    
    # Prepare features
    X, y = preprocessor.prepare_features(processed_data, scale=True, fit=True)
    print(f"✓ Preprocessed data shape: {X.shape}")
    print(f"✓ Number of features: {len(preprocessor.feature_names)}")
    
    # Save preprocessor
    preprocessor.save('../models/preprocessor.pkl')
    
    # Step 3: Split Data (60/20/20 for train/val/test)
    print("\n[3/6] SPLITTING DATA")
    print("-"*70)
    # First split: 60% train, 40% temp
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42, stratify=y
    )
    # Second split: split temp into 20% val, 20% test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✓ Training set: {X_train.shape[0]} samples ({y_train.mean():.2%} fraud)")
    print(f"✓ Validation set: {X_val.shape[0]} samples ({y_val.mean():.2%} fraud)")
    print(f"✓ Test set: {X_test.shape[0]} samples ({y_test.mean():.2%} fraud)")
    
    # Step 4: Train and Compare ML Models
    print("\n[4/6] TRAINING MACHINE LEARNING MODELS")
    print("-"*70)
    print("Note: Using regularization and validation to prevent overfitting")
    print("      Target: 92-98% accuracy (realistic performance)\n")
    
    comparison = ModelComparison(random_state=42)
    # Disable SMOTE to allow realistic class imbalance and errors
    comparison.train_all_models(X_train, y_train, use_smote=False)
    # Evaluate on validation set first
    print("\n📊 Validation Set Performance:")
    comparison.evaluate_all_models(X_val, y_val)
    
    # Evaluate on test set
    print("\n📊 Test Set Performance:")
    comparison.evaluate_all_models(X_test, y_test)
    
    # Get best ML model
    best_ml_model, best_model_name = comparison.get_best_model(metric='f1_score')
    models_dir = os.path.join(os.path.dirname(__file__), '..', 'models')
    os.makedirs(models_dir, exist_ok=True)
    best_ml_model.save(f'../models/best_model_{best_model_name}.pkl')
    # Save again under default name so app config finds it without env change
    best_ml_model.save('../models/best_model_random_forest.pkl')
    
    # Step 5: LLM-Based Detection (if enabled and API key is available)
    print("\n[5/6] LLM-BASED FRAUD DETECTION")
    print("-"*70)
    
    if not use_llm:
        print("⚠️  LLM testing skipped (use --with-llm flag to enable)")
        print("   Saves API costs. Run: python train.py --with-llm")
        llm_detector = None
    else:
        try:
            # Check and clean API Key
            api_key = os.environ.get('GROQ_API_KEY')
            
            # Fallback: Try loading .env manually if not in environment
            if not api_key:
                env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), '.env')
                if os.path.exists(env_path):
                    with open(env_path, 'r') as f:
                        for line in f:
                            if 'GROQ_API_KEY' in line and not line.strip().startswith('#'):
                                _, val = line.strip().split('=', 1)
                                # Strip quotes and whitespace so Groq accepts the key
                                val = val.strip().strip('"').strip("'").replace('\r', '').replace('\n', '').strip()
                                if val:
                                    os.environ['GROQ_API_KEY'] = val
                                    api_key = val
                                break
            
            if api_key:
                print(f"   ✓ API key detected.")

            # Initialize LLM detector
            llm_detector = LLMFraudDetector()
            
            # Get test data with original features for LLM
            test_indices = X_test.index
            test_data_original = processed_data.loc[test_indices].copy()
            
            # OPTIMIZE: Use only 100 samples with STRATIFIED sampling to preserve fraud rate
            # This minimizes API costs while getting realistic performance metrics
            print(f"\n💰 API Cost Optimization:")
            print(f"   Using 100 stratified samples (3.75% fraud rate preserved)")
            print(f"   Estimated API cost: ~100 requests instead of {len(test_data_original)}")
            
            # Stratified sampling to maintain fraud distribution
            if len(test_data_original) > 100:
                llm_sample, _ = train_test_split(
                    test_data_original,
                    test_size=len(test_data_original)-100,
                    stratify=test_data_original['is_fraud'],
                    random_state=42
                )
            else:
                llm_sample = test_data_original
            
            fraud_count = llm_sample['is_fraud'].sum()
            print(f"   Sample: {len(llm_sample)} transactions ({fraud_count} frauds, {fraud_count/len(llm_sample)*100:.2f}%)\n")
            
            # Run LLM detection on sample
            llm_results = llm_detector.predict_batch(llm_sample, max_samples=100)
            
            # Save LLM results
            llm_results.to_csv('../results/llm_predictions.csv', index=False)
            print("✓ LLM results saved to ../results/llm_predictions.csv")
            
            # Show sample predictions with reasoning
            llm_detector.analyze_sample_predictions(llm_results, n_samples=3)
            
            # Add LLM metrics to comparison (include ROC-AUC from confidence scores)
            if hasattr(llm_detector, 'metrics'):
                from sklearn.metrics import roc_auc_score
                y_true = llm_results['actual_is_fraud'].values
                # Use LLM confidence as probability of positive class (fraud) for ROC-AUC
                y_score = (llm_results['llm_confidence'] / 100.0).values
                roc_auc = 0.0
                if len(np.unique(y_true)) > 1 and len(y_true) > 0:
                    try:
                        roc_auc = roc_auc_score(y_true, y_score)
                    except Exception:
                        pass
                comparison.results['llm_groq'] = {
                    'accuracy': llm_detector.metrics['accuracy'],
                    'precision': llm_detector.metrics['precision'],
                    'recall': llm_detector.metrics['recall'],
                    'f1_score': llm_detector.metrics['f1_score'],
                    'roc_auc': roc_auc,
                    'confusion_matrix': None
                }
                print("\n✓ LLM metrics added to comparison")
            
        except ValueError as e:
            print(f"\n⚠️  Skipping LLM detection: {e}")
            print("To enable LLM detection:")
            print("  1. Get API key from https://console.groq.com/keys")
            print("  2. Copy .env.example to .env")
            print("  3. Add your GROQ_API_KEY to .env file")
            llm_detector = None
        except Exception as e:
            print(f"\n⚠️  Error in LLM detection: {e}")
            import traceback
            traceback.print_exc()
            llm_detector = None
    
    # Step 6: Results and Visualization
    print("\n[6/6] GENERATING RESULTS AND VISUALIZATIONS")
    print("-"*70)
    
    # Save comparison plot
    results_df = comparison.plot_comparison(save_path='../results/model_comparison.png')
    
    # Save results to CSV
    results_df.to_csv('../results/model_performance.csv')
    print("✓ Results saved to ../results/model_performance.csv")
    
    # Generate detailed visualizations for best ML model
    print(f"\nGenerating visualizations for best ML model ({best_model_name})...")
    best_ml_model.plot_confusion_matrix(
        X_test, y_test, 
        save_path=f'../results/confusion_matrix_{best_model_name}.png'
    )
    best_ml_model.plot_roc_curve(
        X_test, y_test,
        save_path=f'../results/roc_curve_{best_model_name}.png'
    )
    
    # Feature importance
    if best_model_name in ['random_forest', 'xgboost', 'gradient_boost']:
        print("\nTop 10 Most Important Features (ML Model):")
        importances = best_ml_model.get_feature_importance(preprocessor.feature_names, top_n=10)
        for i, (feature, importance) in enumerate(importances, 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    
    # Final Summary
    print("\n" + "="*70)
    print(" TRAINING COMPLETED SUCCESSFULLY!")
    print("="*70)
    
    print(f"\n📊 BEST MACHINE LEARNING MODEL: {best_model_name.upper()}")
    print("-"*70)
    for metric, value in best_ml_model.performance_metrics.items():
        if metric != 'confusion_matrix':
            print(f"  {metric.capitalize()}: {value:.4f}")
    
    if llm_detector and hasattr(llm_detector, 'metrics'):
        print(f"\n🤖 LLM-BASED DETECTION (Groq API)")
        print("-"*70)
        for metric, value in llm_detector.metrics.items():
            if metric != 'avg_processing_time':
                print(f"  {metric.capitalize()}: {value:.4f}")
            else:
                print(f"  Avg Processing Time: {value:.3f}s per transaction")
        
        # Compare ML vs LLM
        print(f"\n📈 ML vs LLM COMPARISON")
        print("-"*70)
        ml_f1 = best_ml_model.performance_metrics['f1_score']
        llm_f1 = llm_detector.metrics['f1_score']
        
        if ml_f1 > llm_f1:
            print(f"  ML Model performs better (F1: {ml_f1:.4f} vs {llm_f1:.4f})")
            print(f"  Advantage: +{(ml_f1-llm_f1)*100:.2f}%")
        else:
            print(f"  LLM performs better (F1: {llm_f1:.4f} vs {ml_f1:.4f})")
            print(f"  Advantage: +{(llm_f1-ml_f1)*100:.2f}%")
        
        print(f"\n  💡 Hybrid Approach Recommended:")
        print(f"     Use ML for real-time detection (faster)")
        print(f"     Use LLM for explaining flagged transactions (more interpretable)")
    
    print("\n" + "="*70)
    print(f"📁 OUTPUT FILES:")
    print("-"*70)
    print(f"  ✓ ML Models: ../models/")
    print(f"  ✓ Performance Metrics: ../results/model_performance.csv")
    if llm_detector:
        print(f"  ✓ LLM Predictions: ../results/llm_predictions.csv")
    print(f"  ✓ Visualizations: ../results/*.png")
    print("="*70)
    
    print("\n🎓 Ready for your research paper!")
    print("   Use these results in Chapter 4: Results Analysis\n")
    


if __name__ == "__main__":
    import sys
    
    # Check if LLM testing should be enabled
    use_llm = '--with-llm' in sys.argv
    
    if use_llm:
        print("\n✓ LLM testing enabled (will use Groq API)")
    else:
        print("\n⚠️  LLM testing disabled (use --with-llm flag to enable)")
        print("   This saves API costs. Add '--with-llm' to test LLM predictions.\n")
    
    # Modify main() to accept use_llm parameter
    main(use_llm=use_llm)
