"""
Prediction orchestration (SRP).
Orchestrates ML prediction, risk scoring, decision layer, and optional LLM explanation.
"""

from typing import Optional, Any, Callable
import numpy as np

from risk_scoring import RiskScorer, RiskLevel
from decision_layer import DecisionLayer, Decision


def run_risk_based_prediction(
    ml_probability: float,
    risk_scorer: RiskScorer,
    decision_layer: DecisionLayer,
    llm_predictor: Optional[Any] = None,
    transaction_data: Optional[dict] = None,
) -> dict:
    """
    Run risk-based prediction pipeline.

    Args:
        ml_probability: ML fraud probability (0–1).
        risk_scorer: Risk scorer instance.
        decision_layer: Decision layer instance.
        llm_predictor: Optional LLM detector (must have predict_single).
        transaction_data: Transaction dict for LLM (required if invoking LLM).

    Returns:
        Dict with risk_score, risk_level, action, explanation, etc.
    """
    risk_score, risk_level = risk_scorer.score_and_classify(ml_probability)
    decision = decision_layer.decide(risk_level)

    explanation = None
    risk_factors = []

    if decision.invoke_llm and llm_predictor is not None and transaction_data:
        try:
            _, _, reasoning, factors = llm_predictor.predict_single(transaction_data)
            explanation = reasoning
            risk_factors = factors if isinstance(factors, list) else []
        except Exception:
            explanation = "LLM explanation unavailable."

    return {
        "risk_score": risk_score,
        "risk_level": risk_level.value,
        "action": decision.action,
        "explanation": explanation,
        "risk_factors": risk_factors,
        "probability": round(ml_probability, 4),
    }
