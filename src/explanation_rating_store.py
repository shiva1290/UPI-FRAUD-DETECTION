"""
Explanation quality rating persistence (SRP).
Stores and loads human ratings for explanation quality.
"""

import json
import os
from datetime import datetime
from typing import Dict, Any, Optional, List
import numpy as np


class ExplanationRatingStore:
    """Handles storage and retrieval of explanation quality ratings. Single responsibility."""

    @staticmethod
    def save_rating(
        explanation_type: str,  # "shap" or "llm"
        transaction_id: str,
        rating: int,  # 1-5 scale
        comments: Optional[str] = None,
        path: str = "results/explanation_ratings.json",
    ) -> None:
        """Save a single rating."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        ratings = ExplanationRatingStore.load_all(path) or []
        ratings.append(
            {
                "explanation_type": explanation_type,
                "transaction_id": transaction_id,
                "rating": int(rating),
                "comments": comments or "",
                "timestamp": datetime.now().isoformat(),
            }
        )
        with open(path, "w") as f:
            json.dump({"ratings": ratings}, f, indent=2)

    @staticmethod
    def load_all(path: str = "results/explanation_ratings.json") -> Optional[List[Dict[str, Any]]]:
        """Load all ratings."""
        if not os.path.exists(path):
            return []
        try:
            with open(path, "r") as f:
                data = json.load(f)
            return data.get("ratings", [])
        except Exception:
            return []

    @staticmethod
    def get_average_ratings(
        path: str = "results/explanation_ratings.json",
    ) -> Dict[str, float]:
        """Get average ratings per explanation type."""
        ratings = ExplanationRatingStore.load_all(path) or []
        if not ratings:
            return {"shap": 0.0, "llm": 0.0}
        shap_ratings = [r["rating"] for r in ratings if r.get("explanation_type") == "shap"]
        llm_ratings = [r["rating"] for r in ratings if r.get("explanation_type") == "llm"]
        return {
            "shap": float(np.mean(shap_ratings)) if shap_ratings else 0.0,
            "llm": float(np.mean(llm_ratings)) if llm_ratings else 0.0,
        }
