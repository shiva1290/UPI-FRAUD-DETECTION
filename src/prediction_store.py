"""
Prediction store (SRP).
Manages in-memory storage of recent predictions. DIP: swap for DB in production.
"""

from typing import List, Optional, Tuple

# In-memory store (replace with database in production)
_predictions: List[dict] = []
MAX_SIZE = 100


def append(prediction: dict) -> None:
    """Append a prediction and trim if over limit."""
    _predictions.append(prediction)
    while len(_predictions) > MAX_SIZE:
        _predictions.pop(0)


def find_by_id(pred_id: str) -> Tuple[Optional[int], Optional[dict]]:
    """Find prediction by id. Returns (index, pred) or (None, None)."""
    for i, p in enumerate(_predictions):
        if p.get('id') == pred_id:
            return i, p
    return None, None


def get_recent(limit: int = 20) -> List[dict]:
    """Return last N predictions (newest last)."""
    return _predictions[-limit:]


def get_all() -> List[dict]:
    """Return all predictions (for internal use)."""
    return _predictions
