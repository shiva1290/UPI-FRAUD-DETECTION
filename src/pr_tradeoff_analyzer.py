"""
Precision-Recall tradeoff analysis (SRP).
Compares RF and XGBoost across different thresholds.
"""

import numpy as np
from typing import List, Dict, Any
from sklearn.metrics import precision_recall_curve, precision_score, recall_score


class PRTradeoffAnalyzer:
    """
    Analyzes Precision-Recall tradeoffs between models at different thresholds.
    Single responsibility: PR tradeoff computation.
    """

    @staticmethod
    def compare_models(
        y_true: np.ndarray,
        y_scores_rf: np.ndarray,
        y_scores_xgb: np.ndarray,
        thresholds: List[float] = None,
    ) -> Dict[str, Any]:
        """
        Compare RF and XGBoost at multiple thresholds.

        Args:
            y_true: True labels
            y_scores_rf: Random Forest probability scores
            y_scores_xgb: XGBoost probability scores
            thresholds: List of thresholds to evaluate (default: 0.1 to 0.9 in 0.1 steps)

        Returns:
            Dict with precision/recall at each threshold for both models.
        """
        if thresholds is None:
            thresholds = [round(0.1 * i, 2) for i in range(1, 10)]

        results = {
            "thresholds": thresholds,
            "random_forest": [],
            "xgboost": [],
        }

        for thresh in thresholds:
            # RF metrics
            y_pred_rf = (y_scores_rf >= thresh).astype(int)
            rf_precision = float(precision_score(y_true, y_pred_rf, zero_division=0))
            rf_recall = float(recall_score(y_true, y_pred_rf, zero_division=0))

            # XGBoost metrics
            y_pred_xgb = (y_scores_xgb >= thresh).astype(int)
            xgb_precision = float(precision_score(y_true, y_pred_xgb, zero_division=0))
            xgb_recall = float(recall_score(y_true, y_pred_xgb, zero_division=0))

            results["random_forest"].append(
                {"threshold": thresh, "precision": rf_precision, "recall": rf_recall}
            )
            results["xgboost"].append(
                {"threshold": thresh, "precision": xgb_precision, "recall": xgb_recall}
            )

        return results
