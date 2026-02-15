"""
SOLID-friendly abstractions for UPI Fraud Detection.
Defines protocols/interfaces for Dependency Inversion and Liskov Substitution.
"""

from abc import ABC, abstractmethod
from typing import Protocol, Any, Optional, Tuple, List
import numpy as np
import pandas as pd


class FraudDetector(Protocol):
    """Protocol for fraud detectors (ML or LLM). Supports LSP: any implementation is substitutable."""

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict fraud (1) or legitimate (0) for feature matrix."""
        ...

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability of fraud (positive class) for each sample."""
        ...


class MLFraudDetector(FraudDetector, Protocol):
    """Extended protocol for ML models with evaluation capabilities."""

    @property
    def performance_metrics(self) -> dict:
        """Last evaluation metrics."""
        ...

    def train(self, X_train: np.ndarray, y_train: np.ndarray, use_smote: bool = False) -> None:
        """Train the model."""
        ...

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray, threshold: float = 0.5) -> dict:
        """Evaluate and return metrics."""
        ...

    def get_feature_importance(self, feature_names: list, top_n: int = 10) -> Optional[list]:
        """Get top feature importances (for tree-based models)."""
        ...

    def save(self, path: str) -> None:
        """Persist model to disk."""
        ...


class LLMPredictor(Protocol):
    """Protocol for LLM-based fraud prediction. ISP: clients needing only prediction use this."""

    def predict_single(self, transaction_data: dict) -> Tuple[int, float, str, list]:
        """Predict for one transaction. Returns (prediction, confidence, reasoning, risk_factors)."""
        ...

    def predict_batch(self, df: pd.DataFrame, max_samples: int = 100) -> pd.DataFrame:
        """Predict for a batch. Returns DataFrame with predictions."""
        ...


class PreprocessorStep(ABC):
    """Abstract base for preprocessing steps. OCP: add new steps without modifying pipeline."""

    @abstractmethod
    def transform(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Apply transformation. fit=True when processing training data."""
        pass


class ModelFactory(ABC):
    """Abstract factory for ML models. OCP: add new model types by registering, not modifying."""

    @abstractmethod
    def create(self, model_type: str, **kwargs) -> Any:
        """Create a model instance by type."""
        pass
