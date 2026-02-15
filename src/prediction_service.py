"""
Prediction orchestration (SRP).
Orchestrates: feature engineering → ML prediction → risk engine → explanation.
Uses modular components: feature_engineering, ml_prediction, risk_engine, llm_explanation.
"""

from typing import Optional, Any, List

from risk_engine import RiskEngine, RiskAssessment
from llm_explanation import get_explanation


def run_risk_based_prediction(
    ml_probability: float,
    risk_engine: RiskEngine,
    llm_predictor: Optional[Any] = None,
    transaction_data: Optional[dict] = None,
    explanation_generator: Optional[Any] = None,
    feature_order: Optional[List[str]] = None,
    use_llm: bool = False,
) -> dict:
    """
    Run risk-based prediction pipeline with explanations.

    Args:
        ml_probability: ML fraud probability (0–1).
        risk_engine: RiskEngine instance (scoring + thresholds + decision).
        llm_predictor: Optional LLM detector (must have predict_single).
        transaction_data: Transaction dict (required for explanations).
        explanation_generator: Optional rule-based explanation generator.
        feature_order: Top feature names (unused; kept for API compat).
        use_llm: If True, invoke LLM for medium/high. If False, rule-based only (user opts in via UI).

    Returns:
        Dict with risk_score, risk_level, action, explanation, risk_factors, contributing_features.
    """
    assessment = risk_engine.assess(ml_probability)
    should_use_llm = use_llm and assessment.suggest_llm and llm_predictor is not None

    explanation, risk_factors = get_explanation(
        transaction_data=transaction_data or {},
        risk_score=assessment.risk_score,
        use_llm=should_use_llm,
        llm_predictor=llm_predictor,
        rule_generator=explanation_generator,
    )

    contributing_features = risk_factors

    return {
        "risk_score": assessment.risk_score,
        "risk_level": assessment.risk_level,
        "action": assessment.action,
        "explanation": explanation,
        "risk_factors": risk_factors,
        "contributing_features": contributing_features,
        "probability": round(ml_probability, 4),
    }
