"""
Risk scoring and classification (SRP).
Converts ML probability to risk score (0–100) and classifies into Low/Medium/High.
"""

from enum import Enum
from typing import Tuple


class RiskLevel(str, Enum):
    """Risk level classification."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"


class RiskScorer:
    """
    Converts ML fraud probability (0–1) into risk score (0–100)
    and classifies into Low / Medium / High.
    """

    def __init__(
        self,
        low_threshold: float = 30.0,
        medium_threshold: float = 70.0,
    ):
        """
        Args:
            low_threshold: Score below this = Low risk.
            medium_threshold: Score below this (and >= low) = Medium; above = High.
        """
        self.low_threshold = low_threshold
        self.medium_threshold = medium_threshold

    def probability_to_risk_score(self, probability: float) -> float:
        """
        Convert ML probability (0–1, where 1 = fraud) to risk score (0–100).
        Higher score = higher fraud risk.
        """
        return round(float(probability) * 100.0, 2)

    def classify(self, risk_score: float) -> RiskLevel:
        """Classify risk score into Low, Medium, or High."""
        if risk_score < self.low_threshold:
            return RiskLevel.LOW
        if risk_score < self.medium_threshold:
            return RiskLevel.MEDIUM
        return RiskLevel.HIGH

    def score_and_classify(self, probability: float) -> Tuple[float, RiskLevel]:
        """Convert probability to risk score and classify. Returns (risk_score, risk_level)."""
        risk_score = self.probability_to_risk_score(probability)
        risk_level = self.classify(risk_score)
        return risk_score, risk_level
