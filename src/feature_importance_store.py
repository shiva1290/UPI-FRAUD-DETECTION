"""
Feature importance persistence (SRP).
Extract, store, and load feature importance from trained ML models.
"""

import json
import os
from typing import List, Tuple, Optional


class FeatureImportanceStore:
    """Handles storage and retrieval of feature importance. Single responsibility."""

    @staticmethod
    def save(features: List[str], importance: List[float], path: str) -> None:
        """Save feature importance to JSON."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "features": features,
            "importance": [float(x) for x in importance],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"✓ Feature importance saved to {path}")

    @staticmethod
    def load(path: str) -> Optional[Tuple[List[str], List[float]]]:
        """Load feature importance from JSON. Returns (features, importance) or None."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return (data["features"], data["importance"])
        except Exception:
            return None

    @staticmethod
    def extract_from_model(model, feature_names: List[str], top_n: int = 15) -> Optional[Tuple[List[str], List[float]]]:
        """Extract feature importance from trained model. Returns (features, importance) or None."""
        inner = getattr(model, "model", model)
        if not hasattr(inner, "feature_importances_"):
            return None
        imp = inner.feature_importances_
        indices = imp.argsort()[::-1][:top_n]
        names = feature_names if feature_names else [f"feature_{i}" for i in range(len(imp))]
        return (
            [names[i] for i in indices if i < len(names)],
            [float(imp[i]) for i in indices],
        )
