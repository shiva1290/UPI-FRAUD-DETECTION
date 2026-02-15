"""
Model registry for OCP: add new model types without modifying FraudDetectionModel.
"""

from typing import Callable, Dict, Any
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier


def _create_logistic(class_weight: str, random_state: int) -> Any:
    return LogisticRegression(class_weight=class_weight, max_iter=1000, random_state=random_state)


def _create_random_forest(class_weight: str, random_state: int) -> Any:
    return RandomForestClassifier(
        n_estimators=100, class_weight=class_weight, max_depth=10,
        min_samples_split=5, random_state=random_state, n_jobs=-1
    )


def _create_xgboost(class_weight: str, random_state: int) -> Any:
    return XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1, scale_pos_weight=50,
        random_state=random_state, n_jobs=-1
    )


def _create_svm(class_weight: str, random_state: int) -> Any:
    return LinearSVC(
        class_weight=class_weight, dual=False, max_iter=2000, C=0.1,
        random_state=random_state, verbose=0
    )


def _create_gradient_boost(class_weight: str, random_state: int) -> Any:
    return GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.1, max_depth=5, random_state=random_state
    )


# Registry: extend by registering new model types without modifying existing code (OCP)
MODEL_REGISTRY: Dict[str, Callable[[str, int], Any]] = {
    'logistic': _create_logistic,
    'random_forest': _create_random_forest,
    'xgboost': _create_xgboost,
    'svm': _create_svm,
    'gradient_boost': _create_gradient_boost,
}

DEFAULT_MODEL_TYPES = list(MODEL_REGISTRY.keys())


def register_model(model_type: str, factory: Callable[[str, int], Any]) -> None:
    """Register a new model type. OCP: extend without modifying."""
    MODEL_REGISTRY[model_type] = factory


def create_model(model_type: str, class_weight: str = 'balanced', random_state: int = 42) -> Any:
    """Create model instance from registry."""
    factory = MODEL_REGISTRY.get(model_type, MODEL_REGISTRY['random_forest'])
    return factory(class_weight, random_state)
