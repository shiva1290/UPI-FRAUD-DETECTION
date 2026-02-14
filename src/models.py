"""
Machine Learning Models for UPI Fraud Detection
Implements multiple classifiers with class imbalance handling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC, LinearSVC
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.combine import SMOTETomek
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

class FraudDetectionModel:
    def __init__(self, model_type='random_forest', class_weight='balanced', random_state=42):
        """
        Initialize fraud detection model
        
        Args:
            model_type: Type of model ('logistic', 'random_forest', 'xgboost', 'svm', 'gradient_boost')
            class_weight: Strategy for handling class imbalance
            random_state: Random seed
        """
        self.model_type = model_type
        self.class_weight = class_weight
        self.random_state = random_state
        self.model = self._initialize_model()
        self.performance_metrics = {}
        
    def _initialize_model(self):
        """Initialize the specified model"""
        
        models = {
            'logistic': LogisticRegression(
                class_weight=self.class_weight,
                max_iter=1000,
                random_state=self.random_state
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                class_weight=self.class_weight,
                max_depth=10,
                min_samples_split=5,
                random_state=self.random_state,
                n_jobs=-1
            ),
            'xgboost': XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                scale_pos_weight=50,  # Handle imbalance
                random_state=self.random_state,
                n_jobs=-1
            ),
            'svm': LinearSVC(
                class_weight=self.class_weight,
                dual=False,  # Use primal formulation for n_samples > n_features
                max_iter=2000,  # Increase for better convergence
                C=0.1,  # Lower C for better generalization with imbalanced data
                random_state=self.random_state,
                verbose=0
            ),
            'gradient_boost': GradientBoostingClassifier(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=self.random_state
            )
        }
        
        return models.get(self.model_type, models['random_forest'])
    
    def train(self, X_train, y_train, use_smote=False):
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training labels
            use_smote: Whether to use SMOTE for balancing
        """
        
        if use_smote:
            print("Applying SMOTE for class balancing...")
            smote = SMOTE(random_state=self.random_state)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            print(f"After SMOTE - Class distribution: {np.bincount(y_train)}")
        
        print(f"Training {self.model_type} model...")
        self.model.fit(X_train, y_train)
        print("✓ Training completed!")
        
    def predict(self, X):
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        else:
            # For LinearSVC, use decision function and normalize to [0,1]
            decision = self.model.decision_function(X)
            # Normalize to approximate probabilities using sigmoid
            from scipy.special import expit
            return expit(decision)
    
    def evaluate(self, X_test, y_test, threshold=0.5):
        """
        Evaluate model performance
        
        Args:
            X_test: Test features
            y_test: Test labels
            threshold: Classification threshold
        
        Returns:
            Dictionary of performance metrics
        """
        
        # Get predictions
        y_pred_proba = self.predict_proba(X_test)
        y_pred = (y_pred_proba >= threshold).astype(int)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, zero_division=0),
            'recall': recall_score(y_test, y_pred, zero_division=0),
            'f1_score': f1_score(y_test, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_test, y_pred_proba) if len(np.unique(y_test)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        self.performance_metrics = metrics
        
        return metrics
    
    def print_evaluation(self, X_test, y_test):
        """Print detailed evaluation results"""
        
        metrics = self.evaluate(X_test, y_test)
        
        print(f"\n{'='*60}")
        print(f"Model: {self.model_type.upper()}")
        print(f"{'='*60}")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1-Score:  {metrics['f1_score']:.4f}")
        print(f"ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"{'='*60}\n")
        
        # Classification report
        y_pred = self.predict(X_test)
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Legitimate', 'Fraud']))
        
        return metrics
    
    def plot_confusion_matrix(self, X_test, y_test, save_path=None):
        """Plot confusion matrix"""
        
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Legitimate', 'Fraud'],
                   yticklabels=['Legitimate', 'Fraud'])
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        
        plt.show()
    
    def plot_roc_curve(self, X_test, y_test, save_path=None):
        """Plot ROC curve"""
        
        y_pred_proba = self.predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{self.model_type.upper()} (AUC = {roc_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        
        plt.show()
    
    def get_feature_importance(self, feature_names, top_n=10):
        """Get feature importance for tree-based models"""
        
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1][:top_n]
            
            plt.figure(figsize=(10, 6))
            plt.title(f'Top {top_n} Feature Importances - {self.model_type.upper()}')
            plt.bar(range(top_n), importances[indices])
            plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
            plt.ylabel('Importance')
            plt.tight_layout()
            plt.show()
            
            return list(zip([feature_names[i] for i in indices], importances[indices]))
        else:
            print(f"Feature importance not available for {self.model_type}")
            return None
    
    def save(self, path):
        """Save model to disk"""
        joblib.dump(self, path)
        print(f"✓ Model saved to {path}")
    
    @staticmethod
    def load(path):
        """Load model from disk"""
        return joblib.load(path)


class ModelComparison:
    """Compare multiple fraud detection models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.results = {}
        
    def train_all_models(self, X_train, y_train, use_smote=False):
        """Train all model types"""
        
        model_types = ['logistic', 'random_forest', 'xgboost', 'svm', 'gradient_boost']
        
        for model_type in model_types:
            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} Model")
            print(f"{'='*60}")
            
            model = FraudDetectionModel(
                model_type=model_type,
                random_state=self.random_state
            )
            model.train(X_train, y_train, use_smote=use_smote)
            self.models[model_type] = model
    
    def evaluate_all_models(self, X_test, y_test):
        """Evaluate all trained models"""
        
        for model_name, model in self.models.items():
            metrics = model.print_evaluation(X_test, y_test)
            self.results[model_name] = metrics
    
    def plot_comparison(self, save_path=None):
        """Plot comparison of all models"""
        
        metrics_df = pd.DataFrame(self.results).T
        metrics_df = metrics_df.drop('confusion_matrix', axis=1)
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        titles = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        
        for idx, (metric, title) in enumerate(zip(metrics, titles)):
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
        
        plt.show()
        
        return metrics_df
    
    def get_best_model(self, metric='f1_score'):
        """Get the best performing model based on a metric"""
        
        best_model_name = max(self.results, key=lambda x: self.results[x][metric])
        best_score = self.results[best_model_name][metric]
        
        print(f"\n🏆 Best Model: {best_model_name.upper()}")
        print(f"   {metric.upper()}: {best_score:.4f}")
        
        return self.models[best_model_name], best_model_name


if __name__ == "__main__":
    print("Testing Fraud Detection Models...")
    
    # This would typically load preprocessed data
    # For testing purposes, you'd run this after preprocessing
    print("Run train.py to train and evaluate models")
