"""
SHAP explainability utilities (SRP per class).
Provides global and local SHAP explanations for tree-based models.
"""

from typing import Dict, Any, List, Sequence

import numpy as np


class ShapExplainer:
    """
    Computes SHAP values for tree-based classifiers (e.g., RandomForest).
    Single responsibility: SHAP-based explainability (no persistence).
    """

    @staticmethod
    def _get_tree_explainer(inner_model):
        """
        Lazily import shap and build a TreeExplainer.
        Separated to avoid hard dependency at import-time.
        """
        try:
            import shap  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "The 'shap' library is required for SHAP explanations. "
                "Install it with 'pip install shap'."
            ) from exc

        return shap.TreeExplainer(inner_model)

    @staticmethod
    def compute_global(
        inner_model,
        X_background,
        feature_names: Sequence[str],
        max_samples: int = 1000,
    ) -> Dict[str, Any]:
        """
        Compute global SHAP importance as mean |SHAP| per feature.

        Args:
            inner_model: underlying sklearn model (e.g., RandomForestClassifier).
            X_background: 2D array or DataFrame used as background for SHAP.
            feature_names: ordered list of feature names.
            max_samples: max number of background samples to use.
        """
        if X_background is None:
            raise ValueError("X_background cannot be None for SHAP global computation.")

        X_bg = X_background
        # Downsample for performance if necessary
        try:
            # pandas DataFrame path
            if hasattr(X_bg, "sample"):
                n = len(X_bg)
                if n > max_samples:
                    X_bg = X_bg.sample(max_samples, random_state=42)
        except Exception:
            # Fallback: numpy array path
            X_arr = np.asarray(X_bg)
            if X_arr.shape[0] > max_samples:
                idx = np.random.RandomState(42).choice(X_arr.shape[0], size=max_samples, replace=False)
                X_bg = X_arr[idx]

        X_bg_arr = np.asarray(X_bg)
        explainer = ShapExplainer._get_tree_explainer(inner_model)

        # For binary classification, shap_values is list [class0, class1]; use positive class (1)
        shap_values = explainer.shap_values(X_bg_arr)
        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_pos = np.array(shap_values[1])
        else:
            shap_pos = np.array(shap_values)

        mean_abs_shap = np.mean(np.abs(shap_pos), axis=0)
        # Align lengths just in case
        k = min(len(feature_names), len(mean_abs_shap))
        features = list(feature_names)[:k]
        # Leave conversion/robustness to ShapStore; use plain Python scalars
        importance = mean_abs_shap[:k].tolist()

        return {
            "features": features,
            "importance": importance,
        }

    @staticmethod
    def compute_local(
        inner_model,
        X_row,
        feature_names: Sequence[str],
        true_label: int,
    ) -> Dict[str, Any]:
        """
        Compute a local SHAP explanation for a single transaction.

        Args:
            inner_model: underlying sklearn model.
            X_row: 1D or 2D array / Series with a single example.
            feature_names: ordered list of feature names.
            true_label: ground-truth label (0 legitimate, 1 fraud).
        """
        x = X_row
        # Ensure shape (1, n_features)
        if hasattr(x, "values") and getattr(x, "ndim", 1) == 1:
            x_arr = np.asarray(x.values, dtype=float).reshape(1, -1)
        else:
            x_arr = np.asarray(x, dtype=float)
            if x_arr.ndim == 1:
                x_arr = x_arr.reshape(1, -1)

        explainer = ShapExplainer._get_tree_explainer(inner_model)
        shap_values = explainer.shap_values(x_arr)

        if isinstance(shap_values, list) and len(shap_values) > 1:
            shap_pos = np.array(shap_values[1])[0]
            base_value = float(np.atleast_1d(explainer.expected_value)[1])
        else:
            shap_pos = np.array(shap_values)[0]
            base_value = float(np.atleast_1d(explainer.expected_value)[0])

        # Predict probability for this example
        if hasattr(inner_model, "predict_proba"):
            proba = float(inner_model.predict_proba(x_arr)[:, 1][0])
        else:
            from scipy.special import expit

            proba = float(expit(inner_model.decision_function(x_arr))[0])

        k = min(len(feature_names), len(shap_pos))
        names = list(feature_names)[:k]
        shap_vals = shap_pos[:k].tolist()
        values = x_arr[0, :k].tolist()

        # Sort by absolute contribution descending for easier display
        order = np.argsort(np.abs(shap_pos[:k]))[::-1]
        ordered_names = [names[i] for i in order]
        ordered_vals = [float(values[i]) for i in order]
        ordered_shap = [float(shap_vals[i]) for i in order]

        return {
            "feature_names": ordered_names,
            "feature_values": ordered_vals,
            "shap_values": ordered_shap,
            "base_value": base_value,
            "predicted_proba": proba,
            "true_label": int(true_label),
        }

