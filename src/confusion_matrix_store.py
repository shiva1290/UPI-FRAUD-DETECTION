"""
Confusion matrix persistence (SRP).
Saves and loads confusion matrix data for the selected model.
"""

import json
import os
from typing import List, Optional, Tuple


class ConfusionMatrixStore:
    """Handles storage and retrieval of confusion matrix. Single responsibility."""

    @staticmethod
    def save(matrix: List[List[int]], model_name: str, path: str) -> None:
        """Save confusion matrix to JSON. Rows: [TN, FP], [FN, TP]."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        data = {
            "model": model_name,
            "matrix": [[int(c) for c in row] for row in matrix],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load(path: str) -> Optional[Tuple[List[List[int]], str]]:
        """Load confusion matrix from JSON. Returns (matrix, model_name) or None."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return (data["matrix"], data.get("model", "unknown"))
        except Exception:
            return None
