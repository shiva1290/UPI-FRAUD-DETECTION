"""
Decision layer (SRP).
Determines system action and whether to invoke LLM based on risk level.
"""

from typing import NamedTuple

from risk_scoring import RiskLevel


class Decision(NamedTuple):
    """System decision for a transaction."""

    action: str  # allow | review | block
    invoke_llm: bool  # Whether to fetch LLM explanation


class DecisionLayer:
    """
    Determines system action and LLM invocation based on risk level.
    OCP: extend by adding new rules without modifying existing logic.
    """

    def __init__(
        self,
        llm_for_medium: bool = True,
        llm_for_high: bool = True,
    ):
        """
        Args:
            llm_for_medium: Invoke LLM for Medium risk.
            llm_for_high: Invoke LLM for High risk.
        """
        self.llm_for_medium = llm_for_medium
        self.llm_for_high = llm_for_high

    def decide(self, risk_level: RiskLevel) -> Decision:
        """
        Return system action and whether to invoke LLM.

        - Low: allow, no LLM
        - Medium: review, LLM optional
        - High: block, LLM optional
        """
        if risk_level == RiskLevel.LOW:
            return Decision(action="allow", invoke_llm=False)
        if risk_level == RiskLevel.MEDIUM:
            return Decision(action="review", invoke_llm=self.llm_for_medium)
        return Decision(action="block", invoke_llm=self.llm_for_high)
