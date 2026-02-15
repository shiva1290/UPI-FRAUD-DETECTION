"""
Feature Engineering module (SRP).
Prepares raw transaction data for ML prediction.
Handles: cleaning, feature engineering, encoding, scaling via preprocessor.
"""

from typing import Any, Tuple, Union

import numpy as np
import pandas as pd


def prepare_for_ml(
    preprocessor: Any,
    transaction_data: Union[dict, pd.DataFrame],
    fit: bool = False,
) -> Tuple[np.ndarray, Any]:
    """
    Prepare transaction data for ML prediction.

    Pipeline: clean → engineer features → encode → scale → align columns.

    Args:
        preprocessor: UPIPreprocessor instance (handles feature engineering internally).
        transaction_data: Single transaction as dict or DataFrame.
        fit: Whether to fit (use False for inference).

    Returns:
        (X, y): Feature matrix and target (y is None for inference).
    """
    if isinstance(transaction_data, dict):
        df = pd.DataFrame([transaction_data])
    else:
        df = transaction_data.copy()
    processed = preprocessor.preprocess(df, fit=fit)
    X, y = preprocessor.prepare_features(processed, fit=fit)
    return X.values if hasattr(X, 'values') else X, y
