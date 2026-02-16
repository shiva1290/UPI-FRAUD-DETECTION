"""
Precision–Recall curve persistence (SRP).
Stores and loads PR curves for all trained ML models.
"""

import json
import os
from typing import List, Dict, Any, Optional


class PrecisionRecallCurveStore:
    """
    Handles storage and retrieval of precision–recall curves.
    Single responsibility: persistence format for PR curves.
    """

    @staticmethod
    def save(curves: List[Dict[str, Any]], path: str) -> None:
        """
        Save PR curves to JSON.

        Each curve entry should be of the form:
        {
            "model": "<model_name>",
            "precision": [float, ...],
            "recall": [float, ...],
            "pr_auc": float
        }
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        # Ensure plain Python types for JSON
        serializable = []
        for c in curves:
            serializable.append(
                {
                    "model": str(c.get("model", "unknown")),
                    "precision": [float(p) for p in c.get("precision", [])],
                    "recall": [float(r) for r in c.get("recall", [])],
                    "pr_auc": float(c.get("pr_auc", 0.0)),
                }
            )
        with open(path, "w") as f:
            json.dump({"curves": serializable}, f, indent=2)

    @staticmethod
    def load(path: str) -> Optional[Dict[str, Any]]:
        """Load PR curves from JSON. Returns dict or None if missing/invalid."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            # Basic validation
            if not isinstance(data, dict) or "curves" not in data:
                return None
            return data
        except Exception:
            return None

