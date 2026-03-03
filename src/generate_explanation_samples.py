"""
Generate stratified ML+SHAP+LLM explanation samples.

This script:
- Loads the trained ML model and preprocessor.
- Samples 300–500 transactions stratified by fraud label.
- For each *flagged* transaction (Medium/High risk), computes:
  - ML risk score (0–100) and risk level.
  - Top SHAP features (name, value, shap_value) via shap.TreeExplainer.
  - LLM explanation using only risk score + top SHAP features.
- Saves the resulting explanations to ../results/explanation_samples.csv
  so the paper can include SHAP vs LLM side-by-side examples.

The LLM is used *only* for explanations; it does not influence classification.
"""

import os
import json
from typing import List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from dotenv import load_dotenv

from models import FraudDetectionModel
from preprocessor import UPIPreprocessor
from risk_engine import RiskEngine
from llm_detector import LLMFraudDetector


def _load_model_and_preprocessor(base_dir: str):
    models_dir = os.path.join(base_dir, "models")
    model_path = os.path.join(models_dir, "best_model_random_forest.pkl")
    preproc_path = os.path.join(models_dir, "preprocessor.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Run training first.")
    if not os.path.exists(preproc_path):
        raise FileNotFoundError(f"Preprocessor not found at {preproc_path}. Run training first.")

    model: FraudDetectionModel = FraudDetectionModel.load(model_path)
    preprocessor: UPIPreprocessor = UPIPreprocessor.load(preproc_path)
    return model, preprocessor


def _compute_shap_top_features(
    inner_model,
    X_row: np.ndarray,
    feature_names: List[str],
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    import shap  # type: ignore

    explainer = shap.TreeExplainer(inner_model)
    X_row_2d = X_row.reshape(1, -1)
    shap_values = explainer.shap_values(X_row_2d)

    if isinstance(shap_values, list) and len(shap_values) > 1:
        shap_pos = np.array(shap_values[1])[0]
    else:
        shap_pos = np.array(shap_values)[0]

    indices = np.argsort(np.abs(shap_pos))[-top_n:][::-1]
    top_features: List[Dict[str, Any]] = []
    for idx in indices:
        top_features.append(
            {
                "name": str(feature_names[idx]),
                "value": float(X_row[idx]),
                "shap_value": float(shap_pos[idx]),
            }
        )
    return top_features


def main(
    target_samples: int = 400,
    min_risk_score: float = 30.0,
    output_path: str = "../results/explanation_samples.csv",
):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    # Load .env so GROQ_API_KEY is available
    load_dotenv(os.path.join(base_dir, ".env"))

    # Load data
    data_path = os.path.join(base_dir, "data", "upi_transactions.csv")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset not found at {data_path}. Run train.py first.")

    df = pd.read_csv(data_path, parse_dates=["timestamp"])
    if "is_fraud" not in df.columns:
        raise ValueError("Dataset must contain 'is_fraud' column.")

    # Load model + preprocessor and risk engine
    model, preprocessor = _load_model_and_preprocessor(base_dir)
    inner_model = getattr(model, "model", model)

    risk_engine = RiskEngine(
        low_threshold=30.0,
        medium_threshold=70.0,
        llm_for_medium=True,
        llm_for_high=True,
    )

    # Initialize LLM explanation helper
    llm = LLMFraudDetector()

    # Preprocess data
    processed = preprocessor.preprocess(df, fit=False)
    X_all, y_all = preprocessor.prepare_features(processed, fit=False)
    feature_names = preprocessor.feature_names or list(X_all.columns)

    # Sample 300–500 rows stratified by fraud label
    total = len(df)
    sample_size = min(max(target_samples, 300), 500)
    if total <= sample_size:
        sample_idx = df.index
    else:
        _, sample_df = train_test_split(
            df,
            test_size=sample_size / total,
            stratify=df["is_fraud"],
            random_state=42,
        )
        sample_idx = sample_df.index

    # Build explanation samples for *flagged* (Medium/High) transactions only
    rows: List[Dict[str, Any]] = []
    used_indices = set()

    for idx in sample_idx:
        if idx in used_indices:
            continue

        x_row = X_all.loc[idx].values
        proba = float(model.predict_proba(x_row.reshape(1, -1))[0])
        assessment = risk_engine.assess(proba)
        risk_score = assessment.risk_score

        # Only keep flagged / review-worthy transactions
        if risk_score < min_risk_score:
            continue

        used_indices.add(idx)

        # SHAP top features
        top_features = _compute_shap_top_features(inner_model, x_row, feature_names, top_n=5)

        # LLM explanation (risk_score + SHAP features only)
        reasoning, risk_factors = llm.explain(risk_score=risk_score, top_features=top_features)

        src_row = df.loc[idx]
        rows.append(
            {
                "index": int(idx),
                "is_fraud": int(src_row["is_fraud"]),
                "risk_score": float(risk_score),
                "risk_level": assessment.risk_level,
                "action": assessment.action,
                "shap_top_features": json.dumps(top_features, ensure_ascii=False),
                "llm_explanation": reasoning,
                "llm_risk_factors": json.dumps(risk_factors, ensure_ascii=False),
            }
        )

        if len(rows) >= sample_size:
            break

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    print(f"✓ Saved {len(out_df)} explanation samples to {output_path}")


if __name__ == "__main__":
    main()

