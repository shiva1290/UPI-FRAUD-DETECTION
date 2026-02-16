"""
SHAP persistence utilities (SRP).
Stores and loads global and local SHAP explanations.
"""

import json
import os
from typing import Dict, Any, Optional


class ShapStore:
    """Handles JSON persistence for SHAP outputs. Single responsibility: storage format."""

    @staticmethod
    def save_global(data: Dict[str, Any], path: str) -> None:
        """
        Save global SHAP importance to JSON.
        Expected data: {"features": [...], "importance": [...]}
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        raw_features = data.get("features", [])
        raw_importance = data.get("importance", [])

        importance: list = []
        for v in raw_importance:
            try:
                # Common case: scalar number
                importance.append(float(v))
            except (TypeError, ValueError):
                # If we accidentally get a list/array, fall back to its first element
                try:
                    if isinstance(v, (list, tuple)) and v:
                        importance.append(float(v[0]))
                    else:
                        importance.append(0.0)
                except Exception:
                    importance.append(0.0)

        payload = {
            "features": [str(f) for f in raw_features],
            "importance": importance,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load_global(path: str) -> Optional[Dict[str, Any]]:
        """Load global SHAP importance from JSON."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "features" not in data or "importance" not in data:
                return None
            return data
        except Exception:
            return None

    @staticmethod
    def save_local(example: Dict[str, Any], path: str) -> None:
        """
        Save a local SHAP explanation for a single transaction.
        Expected keys: feature_names, feature_values, shap_values, base_value, predicted_proba, true_label.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "feature_names": [str(f) for f in example.get("feature_names", [])],
            "feature_values": [float(v) for v in example.get("feature_values", [])],
            "shap_values": [float(s) for s in example.get("shap_values", [])],
            "base_value": float(example.get("base_value", 0.0)),
            "predicted_proba": float(example.get("predicted_proba", 0.0)),
            "true_label": int(example.get("true_label", 0)),
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load_local(path: str) -> Optional[Dict[str, Any]]:
        """Load a local SHAP explanation from JSON."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "feature_names" not in data or "shap_values" not in data:
                return None
            return data
        except Exception:
            return None

