"""
Model visualization (SRP: single responsibility for plotting).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server / training runs
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score


class ModelVisualizer:
    """Handles all ML model plotting. Single responsibility: visualization."""

    @staticmethod
    def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str, save_path: str = None) -> None:
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Legitimate', 'Fraud'],
                    yticklabels=['Legitimate', 'Fraud'])
        plt.title(f'Confusion Matrix - {model_name.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_roc_curve(y_true: np.ndarray, y_pred_proba: np.ndarray,
                      model_name: str, save_path: str = None) -> None:
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'{model_name.upper()} (AUC = {roc_auc:.3f})', linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ ROC curve saved to {save_path}")
        plt.close()

    @staticmethod
    def plot_feature_importance(importances: np.ndarray, feature_names: list,
                               model_name: str, top_n: int = 10) -> None:
        """Plot feature importance bar chart."""
        indices = np.argsort(importances)[::-1][:top_n]
        plt.figure(figsize=(10, 6))
        plt.title(f'Top {top_n} Feature Importances - {model_name.upper()}')
        plt.bar(range(top_n), importances[indices])
        plt.xticks(range(top_n), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.ylabel('Importance')
        plt.tight_layout()
        # Do not call plt.show() in backend/server context
        plt.close()
