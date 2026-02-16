"""
Explanation comparison utilities (SRP).
Compares SHAP and LLM explanations for the same transaction.
"""

from typing import Dict, Any, Optional, List
import numpy as np


class ExplanationComparator:
    """
    Compares SHAP and LLM explanations for the same transaction.
    Single responsibility: comparison logic (no persistence).
    """

    @staticmethod
    def compare_explanations(
        shap_data: Dict[str, Any],
        llm_reasoning: str,
        llm_risk_factors: List[str],
        transaction_data: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Compare SHAP and LLM explanations for the same transaction.

        Args:
            shap_data: Local SHAP explanation dict (feature_names, shap_values, etc.)
            llm_reasoning: LLM natural-language explanation text
            llm_risk_factors: LLM risk factors list
            transaction_data: Original transaction data

        Returns:
            Comparison dict with alignment metrics and side-by-side view.
        """
        shap_features = shap_data.get("feature_names", [])
        shap_values = shap_data.get("shap_values", [])
        shap_top = [
            {"feature": f, "shap_value": float(shap_values[i])}
            for i, f in enumerate(shap_features[:10])
        ]

        # Extract feature mentions from LLM reasoning
        llm_mentioned_features = ExplanationComparator._extract_feature_mentions(
            llm_reasoning, llm_risk_factors
        )

        # Compute alignment: how many top SHAP features are mentioned by LLM?
        shap_top_names = {f["feature"].lower() for f in shap_top[:5]}
        llm_mentioned_lower = {f.lower() for f in llm_mentioned_features}
        alignment_count = len(shap_top_names & llm_mentioned_lower)
        alignment_score = alignment_count / max(len(shap_top_names), 1)

        return {
            "shap_explanation": {
                "top_features": shap_top,
                "predicted_proba": shap_data.get("predicted_proba", 0.0),
                "base_value": shap_data.get("base_value", 0.0),
            },
            "llm_explanation": {
                "reasoning": llm_reasoning,
                "risk_factors": llm_risk_factors,
                "mentioned_features": llm_mentioned_features,
            },
            "comparison": {
                "alignment_score": float(alignment_score),
                "alignment_count": alignment_count,
                "shap_top_count": len(shap_top_names),
            },
            "transaction_data": transaction_data,
        }

    @staticmethod
    def _extract_feature_mentions(reasoning: str, risk_factors: List[str]) -> List[str]:
        """Extract feature names mentioned in LLM reasoning."""
        feature_keywords = {
            "amount": ["amount", "transaction amount", "value"],
            "transaction_velocity": ["velocity", "frequency", "rapid", "multiple transactions"],
            "device_change": ["device", "device change", "new device"],
            "location_change_km": ["location", "location change", "distance"],
            "failed_attempts": ["failed", "failure", "attempt"],
            "beneficiary_fan_in": ["beneficiary", "fan-in", "recipient"],
            "is_new_beneficiary": ["new beneficiary", "first time"],
            "amount_deviation_pct": ["deviation", "unusual amount", "abnormal"],
        }
        text = (reasoning + " " + " ".join(risk_factors)).lower()
        mentioned = []
        for feat, keywords in feature_keywords.items():
            if any(kw in text for kw in keywords):
                mentioned.append(feat)
        return mentioned
