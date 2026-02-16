"""
Cost-sensitive evaluation utilities (SRP).
Defines FN/FP costs, expected financial loss, and cost–threshold curves.
"""

import json
import os
from typing import Dict, Any, List, Optional

import numpy as np


# Default cost assumptions (can be referenced from README/report)
# Units are arbitrary (e.g., INR); only the ratio matters for comparison.
DEFAULT_FP_COST = 1_000.0   # customer friction, support cost, lost revenue
DEFAULT_FN_COST = 10_000.0  # direct fraud loss, chargebacks, reputational risk


def compute_expected_loss(
    cm: np.ndarray,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
    total_samples: int = 1,
) -> Dict[str, float]:
    """
    Compute expected financial loss given a confusion matrix and costs.

    cm layout (sklearn): [[TN, FP], [FN, TP]]
    """
    if cm is None or cm.size != 4:
        return {"total_cost": 0.0, "cost_per_txn": 0.0, "cost_per_1000": 0.0}

    tn, fp, fn, tp = cm.ravel()
    total_cost = fp_cost * fp + fn_cost * fn
    total_samples = float(total_samples) if total_samples > 0 else 1.0
    cost_per_txn = total_cost / total_samples
    cost_per_1000 = cost_per_txn * 1000.0
    return {
        "total_cost": float(total_cost),
        "cost_per_txn": float(cost_per_txn),
        "cost_per_1000": float(cost_per_1000),
    }


def build_cost_vs_threshold_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    fp_cost: float = DEFAULT_FP_COST,
    fn_cost: float = DEFAULT_FN_COST,
    num_points: int = 50,
) -> Dict[str, List[float]]:
    """
    Compute total cost (per 1,000 transactions) as a function of decision threshold.

    Returns a dict with lists: thresholds, total_cost_per_1000, fp_rate, fn_rate.
    """
    if num_points <= 1:
        num_points = 2

    thresholds = np.linspace(0.0, 1.0, num_points)
    total_cost_per_1000: List[float] = []
    fp_rate: List[float] = []
    fn_rate: List[float] = []

    y_true = np.asarray(y_true)
    y_scores = np.asarray(y_scores)
    n = float(len(y_true)) if len(y_true) > 0 else 1.0

    for t in thresholds:
        y_pred = (y_scores >= t).astype(int)
        # Confusion matrix: [[TN, FP], [FN, TP]]
        tn = float(((y_true == 0) & (y_pred == 0)).sum())
        fp = float(((y_true == 0) & (y_pred == 1)).sum())
        fn = float(((y_true == 1) & (y_pred == 0)).sum())
        tp = float(((y_true == 1) & (y_pred == 1)).sum())

        cost = fp_cost * fp + fn_cost * fn
        cost_per_txn = cost / n
        total_cost_per_1000.append(float(cost_per_txn * 1000.0))
        fp_rate.append(fp / n)
        fn_rate.append(fn / n)

    return {
        "thresholds": [float(t) for t in thresholds],
        "total_cost_per_1000": total_cost_per_1000,
        "fp_rate": fp_rate,
        "fn_rate": fn_rate,
        "fp_cost": float(fp_cost),
        "fn_cost": float(fn_cost),
    }


class CostCurveStore:
    """Handles persistence of cost–threshold curves (SRP)."""

    @staticmethod
    def save(curve: Dict[str, Any], model_name: str, path: str) -> None:
        """Save a single model's cost–threshold curve to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        payload = {
            "model": model_name,
            **curve,
        }
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)

    @staticmethod
    def load(path: str) -> Optional[Dict[str, Any]]:
        """Load a cost–threshold curve from JSON. Returns dict or None."""
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r") as f:
                data = json.load(f)
            if not isinstance(data, dict) or "thresholds" not in data:
                return None
            return data
        except Exception:
            return None

