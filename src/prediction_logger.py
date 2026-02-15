"""
Prediction logger (SRP).
Persists prediction logs for fraud pattern analysis. DIP: swap implementation for DB.
"""

import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Optional


class PredictionLogWriter(ABC):
    """Abstract interface for prediction logging. OCP: add new writers without changing callers."""

    @abstractmethod
    def log(self, prediction: dict) -> None:
        """Persist a prediction log entry."""
        pass


class FilePredictionLogger(PredictionLogWriter):
    """Writes prediction logs to JSONL file for fraud pattern analysis."""

    def __init__(self, log_dir: str = "logs", filename: str = "prediction_logs.jsonl"):
        self.log_dir = log_dir
        self.filename = filename
        os.makedirs(log_dir, exist_ok=True)

    def _sanitize_for_log(self, pred: dict) -> dict:
        """Extract fields suitable for analysis; omit full transaction payload for brevity."""
        td = pred.get("transaction_data") or {}
        return {
            "id": pred.get("id"),
            "timestamp": pred.get("timestamp") or datetime.now().isoformat(),
            "risk_score": pred.get("risk_score"),
            "risk_level": pred.get("risk_level"),
            "action": pred.get("action"),
            "model": pred.get("model"),
            "ml_latency_ms": pred.get("ml_latency_ms"),
            "llm_latency_ms": pred.get("llm_latency_ms"),
            "llm_analyzed": pred.get("llm_analyzed", False),
            "amount": td.get("amount"),
            "hour": td.get("hour"),
            "day_of_week": td.get("day_of_week"),
            "transaction_velocity": td.get("transaction_velocity"),
            "device_change": td.get("device_change"),
            "location_change_km": td.get("location_change_km"),
        }

    def log(self, prediction: dict) -> None:
        path = os.path.join(self.log_dir, self.filename)
        entry = self._sanitize_for_log(prediction)
        try:
            with open(path, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except OSError:
            pass
