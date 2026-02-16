"""
Concept drift simulation (SRP).
Simulates performance degradation over time as data distribution changes.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


class ConceptDriftSimulator:
    """
    Simulates concept drift by gradually shifting feature distributions.
    Single responsibility: drift simulation and performance tracking.
    """

    @staticmethod
    def simulate_drift(
        X_original: np.ndarray,
        y_original: np.ndarray,
        model,
        feature_names: List[str],
        num_periods: int = 10,
        drift_strength: float = 0.1,
    ) -> List[Dict[str, Any]]:
        """
        Simulate concept drift over multiple time periods.

        Args:
            X_original: Original training/test features
            y_original: Original labels
            model: Trained model to evaluate
            feature_names: Feature names
            num_periods: Number of time periods to simulate
            drift_strength: How much to shift distributions per period (0.0-1.0)

        Returns:
            List of performance metrics per period.
        """
        results = []
        # Convert to numpy array if DataFrame
        if isinstance(X_original, pd.DataFrame):
            X_current = X_original.values.copy()
        else:
            X_current = np.array(X_original).copy()

        # Identify key features that might drift (e.g., amount, velocity)
        drift_features = [
            i
            for i, name in enumerate(feature_names)
            if any(kw in name.lower() for kw in ["amount", "velocity", "location"])
        ]

        for period in range(num_periods):
            # Gradually shift feature distributions
            if period > 0:
                for feat_idx in drift_features:
                    if feat_idx < X_current.shape[1]:
                        # Shift mean and add noise
                        feature_values = X_current[:, feat_idx]
                        shift = drift_strength * period * np.std(feature_values)
                        X_current[:, feat_idx] = feature_values + shift + np.random.normal(
                            0, drift_strength * 0.1, size=X_current.shape[0]
                        )

            # Use the model wrapper's predict_proba if available (handles format conversion)
            # Otherwise use inner model directly
            if hasattr(model, "predict_proba"):
                # Model wrapper (FraudDetectionModel) - use its predict_proba method
                y_pred_proba = model.predict_proba(X_current)
            else:
                # Direct sklearn model - get inner model if wrapped
                inner_model = getattr(model, "model", model)
                
                # Convert to DataFrame if needed (sklearn models can handle both)
                if isinstance(X_current, pd.DataFrame):
                    X_pred = X_current
                else:
                    # Try to create DataFrame with feature names if available
                    if feature_names and len(feature_names) >= X_current.shape[1]:
                        X_pred = pd.DataFrame(X_current, columns=feature_names[:X_current.shape[1]])
                    else:
                        X_pred = X_current
                
                if hasattr(inner_model, "predict_proba"):
                    proba_output = inner_model.predict_proba(X_pred)
                    # Handle both 1D and 2D outputs
                    if proba_output.ndim == 2 and proba_output.shape[1] > 1:
                        y_pred_proba = proba_output[:, 1]
                    else:
                        y_pred_proba = proba_output.flatten() if proba_output.ndim > 1 else proba_output
                elif hasattr(inner_model, "decision_function"):
                    y_pred_proba = inner_model.decision_function(X_pred)
                    if y_pred_proba.ndim > 1:
                        y_pred_proba = y_pred_proba.flatten()
                else:
                    y_pred = inner_model.predict(X_pred)
                    y_pred_proba = y_pred.astype(float)
            
            y_pred = (y_pred_proba >= 0.5).astype(int)

            metrics = {
                "period": period,
                "accuracy": float(accuracy_score(y_original, y_pred)),
                "precision": float(precision_score(y_original, y_pred, zero_division=0)),
                "recall": float(recall_score(y_original, y_pred, zero_division=0)),
                "f1_score": float(f1_score(y_original, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_original, y_pred_proba)) if len(np.unique(y_original)) > 1 else 0.0,
            }
            results.append(metrics)

        return results
