"""
ML Prediction module (SRP).
Runs fraud probability prediction using the trained model.
"""

from typing import Any


def predict_fraud_probability(
    model: Any,
    preprocessor: Any,
    transaction_data: dict,
) -> float:
    """
    Predict fraud probability for a transaction.

    Args:
        model: Trained ML model with predict_proba (positive class = fraud).
        preprocessor: UPIPreprocessor instance.
        transaction_data: Validated transaction dict.

    Returns:
        Fraud probability in [0, 1].
    """
    from feature_engineering import prepare_for_ml

    X, _ = prepare_for_ml(preprocessor, transaction_data, fit=False)
    proba = model.predict_proba(X)[0]
    # Positive class (index 1) = fraud
    if hasattr(proba, '__len__') and len(proba) > 1:
        return float(proba[1])
    return float(proba)
