"""
Risk Engine (SRP).
Explicit risk scoring and threshold layer for UPI fraud detection.
Converts ML probability into risk score, classifies risk level, and determines system action.
This is a risk assessment framework, not a binary classifier.
"""

from dataclasses import dataclass
from typing import Tuple

from risk_scoring import RiskScorer, RiskLevel
from decision_layer import DecisionLayer, Decision


@dataclass
class RiskAssessment:
    """Result of risk engine assessment."""

    risk_score: float
    risk_level: str
    action: str
    low_threshold: float
    medium_threshold: float
    suggest_llm: bool


class RiskEngine:
    """
    Explicit risk scoring engine.
    Orchestrates: probability → risk score → risk level → decision.
    OCP: extend by injecting custom scorers or decision rules.
    """

    def __init__(
        self,
        low_threshold: float = 30.0,
        medium_threshold: float = 70.0,
        llm_for_medium: bool = True,
        llm_for_high: bool = True,
    ):
        """
        Args:
            low_threshold: Score below this = Low risk.
            medium_threshold: Score below this (and >= low) = Medium; above = High.
            llm_for_medium: Whether LLM explanation is suggested for Medium risk.
            llm_for_high: Whether LLM explanation is suggested for High risk.
        """
        self.risk_scorer = RiskScorer(
            low_threshold=low_threshold,
            medium_threshold=medium_threshold,
        )
        self.decision_layer = DecisionLayer(
            llm_for_medium=llm_for_medium,
            llm_for_high=llm_for_high,
        )
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold

    def assess(self, ml_probability: float) -> RiskAssessment:
        """
        Assess risk from ML fraud probability (0–1).

        Returns:
            RiskAssessment with score, level, action, and thresholds used.
        """
        risk_score, risk_level = self.risk_scorer.score_and_classify(ml_probability)
        decision = self.decision_layer.decide(risk_level)
        return RiskAssessment(
            risk_score=risk_score,
            risk_level=risk_level.value,
            action=decision.action,
            low_threshold=self.low_threshold,
            medium_threshold=self.medium_threshold,
            suggest_llm=decision.invoke_llm,
        )

    def score_and_classify(self, probability: float) -> Tuple[float, RiskLevel]:
        """Convenience: raw score and level."""
        return self.risk_scorer.score_and_classify(probability)
