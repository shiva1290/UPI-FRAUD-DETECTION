"""
LLM Explanation module (SRP).
Generates human-readable explanations: rule-based or LLM-based.
"""

from typing import Any, List, Optional, Tuple


def get_explanation(
    transaction_data: dict,
    risk_score: float,
    use_llm: bool,
    llm_predictor: Optional[Any],
    rule_generator: Optional[Any],
) -> Tuple[str, List[str]]:
    """
    Get explanation and risk factors for a transaction.

    Args:
        transaction_data: Transaction dict.
        risk_score: Risk score (0–100).
        use_llm: Whether to invoke LLM (user must opt in).
        llm_predictor: LLM detector with predict_single (optional).
        rule_generator: ExplanationGenerator for rule-based fallback.

    Returns:
        (explanation_text, risk_factors)
    """
    explanation = None
    risk_factors = []

    if use_llm and llm_predictor is not None:
        try:
            _, _, reasoning, factors = llm_predictor.predict_single(transaction_data)
            explanation = reasoning
            risk_factors = factors if isinstance(factors, list) else []
        except Exception:
            explanation = "LLM explanation unavailable."

    if not explanation and rule_generator and transaction_data:
        explanation, risk_factors = rule_generator.generate(transaction_data, risk_score)

    if not explanation:
        explanation = f"Risk score: {risk_score:.1f}. No explanation available."
        risk_factors = risk_factors or [f"Risk score {risk_score:.0f}"]

    return explanation, risk_factors
