"""
Machine Learning Models for UPI Fraud Risk Assessment.
Uses model registry (OCP) and visualizer (SRP).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    average_precision_score,
)
from imblearn.over_sampling import SMOTE
import joblib

import matplotlib
matplotlib.use("Agg")  # Use non-interactive backend to avoid Tkinter issues
import matplotlib.pyplot as plt
from model_registry import create_model, DEFAULT_MODEL_TYPES
from model_visualizer import ModelVisualizer


class FraudDetectionModel:
    """ML fraud detector. Implements FraudDetector protocol (LSP)."""

    def __init__(self, model_type='random_forest', class_weight='balanced', random_state=42):
        self.model_type = model_type
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = create_model(model_type, class_weight, random_state)
        self.performance_metrics = {}

    def train(self, X_train, y_train, use_smote=False):
        """Train the model."""
        import time
        train_start = time.time()
        
        if use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Class distribution: {np.bincount(y_train)}")

        print(f"Training {self.model_type} model...")
        if len(X_train) > 100000:
            print(f"  Large dataset ({len(X_train):,} samples) - this may take a few minutes...")
        
        self.model.fit(X_train, y_train)
        
        train_time = time.time() - train_start
        print(f"✓ Training completed in {train_time:.2f} seconds ({train_time/60:.2f} minutes)!")

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def predict_proba(self, X):
        """Get prediction probabilities."""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        from scipy.special import expit
        return expit(self.model.decision_function(X))

    def evaluate(self, X_test, y_test, threshold=0.5):
        """Evaluate model performance."""
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        # Core scalar metrics (used in CSV/model comparison)
        roc_auc = roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        pr_auc = average_precision_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            # ROC-AUC: discrimination ability across thresholds
            'roc_auc': roc_auc,
            # PR-AUC (average precision): focuses on fraud (positive) class; better for imbalance
            'pr_auc': pr_auc,
            'confusion_matrix': confusion_matrix(y_test, y_pred),
        }
        self.performance_metrics = metrics
        return metrics

    def print_evaluation(self, X_test, y_test):
        """Print detailed evaluation results."""
        metrics = self.evaluate(X_test, y_test)
        print(f"\n{'='*60}")
        print(f"Model: {self.model_type.upper()}")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:\n{metrics['confusion_matrix']}")
        print(f"{'='*60}\n")
        y_pred = self.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        return metrics

    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """Plot confusion matrix (delegates to ModelVisualizer)."""
        y_pred = self.predict(X_test)
        ModelVisualizer.plot_confusion_matrix(
            y_test, y_pred, self.model_type, save_path
        )

    def plot_roc_curve(self, X_test, y_test, save_path=None):
        """Plot ROC curve (delegates to ModelVisualizer)."""
        y_pred_proba = self.predict_proba(X_test)
        ModelVisualizer.plot_roc_curve(
            y_test, y_pred_proba, self.model_type, save_path
        )

    def get_feature_importance(self, feature_names, top_n=10):
        """Get feature importance for tree-based models."""
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            return list(zip([feature_names[i] for i in indices], importances[indices]))
        print(f"Feature importance not available for {self.model_type}")
        return None

    def save(self, path):
        """Save model to disk."""
        joblib.dump(self, path)
        print(f"✓ Model saved to {path}")

    @staticmethod
    def load(path):
        """Load model from disk."""
        return joblib.load(path)


class ModelComparison:
    """Compare multiple fraud detection models. Uses DEFAULT_MODEL_TYPES (OCP)."""

    def __init__(self, random_state=42, model_types=None):
        self.random_state = random_state
        self.model_types = model_types or DEFAULT_MODEL_TYPES
        self.models = {}
        self.results = {}

    def train_all_models(self, X_train, y_train, use_smote=False):
        """Train all registered model types."""
        for model_type in self.model_types:
            print(f"\n{'='*60}\nTraining {model_type.upper()} Model\n{'='*60}")
            model = FraudDetectionModel(model_type=model_type, random_state=self.random_state)
            model.train(X_train, y_train, use_smote=use_smote)
            self.models[model_type] = model

    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models."""
        for model_name, model in self.models.items():
            metrics = model.print_evaluation(X_test, y_test)
            self.results[model_name] = metrics

    def plot_comparison(self, save_path=None):
        """Plot comparison of all models."""
        metrics_df = pd.DataFrame(self.results).T
        metrics_df = metrics_df.drop('confusion_matrix', axis=1, errors='ignore')

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        for idx, (metric, title) in enumerate(zip(
            ['accuracy', 'precision', 'recall', 'f1_score'],
            ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        )):
            ax = axes[idx // 2, idx % 2]
            metrics_df[metric].plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
            ax.set_title(title, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(axis='y', alpha=0.3)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Comparison plot saved to {save_path}")
        # Do not call plt.show() in non-interactive / training context
        plt.close(fig)
        return metrics_df

    def get_best_model(self, metric='f1_score'):
        """Get the best performing model based on a metric."""
        best_model_name = max(self.results, key=lambda x: self.results[x][metric])
        best_score = self.results[best_model_name][metric]
        print(f"\n🏆 Best Model: {best_model_name.upper()}\n   {metric.upper()}: {best_score:.4f}")
        return self.models[best_model_name], best_model_name
