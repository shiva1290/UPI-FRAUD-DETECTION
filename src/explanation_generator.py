"""
Rule-based explanation generator (SRP).
Produces human-readable text explaining why a transaction is considered risky.
Aligns with model features: amount, velocity, device change, etc.
"""

from typing import List, Tuple, Optional

# Suspicious thresholds per feature (aligned with model features)
SUSPICIOUS_THRESHOLDS = {
    "amount_deviation_pct": (80, "high amount deviation"),
    "transaction_velocity": (5, "elevated transaction velocity"),
    "device_change": (1, "device change detected"),
    "location_change_km": (50, "significant location change"),
    "failed_attempts": (1, "failed payment attempts"),
    "is_night": (1, "transaction at night"),
    "beneficiary_fan_in": (15, "high beneficiary fan-in"),
    "is_new_beneficiary": (1, "new beneficiary"),
    "reversed_attempts": (1, "reversal attempts"),
    "is_high_amount": (1, "high amount"),
    "amount": (20000, "high transaction amount"),
    "approval_delay_sec": (10, "short approval delay (suspicious)"),
}

# Human-readable feature names
FEATURE_LABELS = {
    "amount_deviation_pct": "Amount deviation",
    "transaction_velocity": "Transaction velocity",
    "device_change": "Device change",
    "location_change_km": "Location change (km)",
    "failed_attempts": "Failed attempts",
    "is_night": "Night transaction",
    "beneficiary_fan_in": "Beneficiary fan-in",
    "is_new_beneficiary": "New beneficiary",
    "reversed_attempts": "Reversal attempts",
    "is_high_amount": "High amount flag",
    "amount": "Amount",
    "approval_delay_sec": "Approval delay",
}


class ExplanationGenerator:
    """
    Generates explanation text from transaction data and feature importance.
    Uses rule-based thresholds aligned with model features.
    """

    def __init__(self, feature_order: Optional[List[str]] = None):
        """
        Args:
            feature_order: Top features by importance (used to rank contributing factors).
        """
        self.feature_order = feature_order or list(SUSPICIOUS_THRESHOLDS.keys())

    def generate(self, transaction_data: dict, risk_score: float) -> Tuple[str, List[str]]:
        """
        Generate explanation and list of contributing risk factors.

        Returns:
            (explanation_text, risk_factors)
        """
        factors = self._identify_contributing_factors(transaction_data)
        # Always show risk-level context for low/medium; add fallback factor when none found
        if not factors:
            if risk_score >= 70:
                fallback = [("Risk level", f"Score {risk_score:.0f} indicates high risk")]
            elif risk_score >= 30:
                fallback = [("Risk level", f"Score {risk_score:.0f} indicates medium risk - consider reviewing")]
            else:
                return (
                    f"Risk score: {risk_score:.1f}. No strong fraud indicators detected. Transaction appears within normal patterns.",
                    [f"Risk score {risk_score:.0f} - Low risk"],
                )
        else:
            fallback = []
        all_factors = factors or fallback
        parts = [f"Risk score: {risk_score:.1f}. Contributing factors:"]
        for i, (label, detail) in enumerate(all_factors[:5], 1):
            parts.append(f"  {i}. {label}: {detail}")
        explanation = "\n".join(parts)
        risk_factors = [f"{label} - {detail}" for label, detail in all_factors]
        return explanation, risk_factors

    def _identify_contributing_factors(self, data: dict) -> List[Tuple[str, str]]:
        """Identify which features contribute to risk, ordered by importance."""
        factors = []
        for feat in self.feature_order:
            if feat not in data:
                continue
            val = data.get(feat)
            if val is None:
                continue
            thr, desc = SUSPICIOUS_THRESHOLDS.get(feat, (None, None))
            if thr is None:
                continue
            label = FEATURE_LABELS.get(feat, feat)
            # Numeric threshold
            if isinstance(thr, (int, float)):
                if isinstance(val, (int, float)):
                    if feat in ("device_change", "is_night", "is_new_beneficiary", "is_high_amount"):
                        if val >= thr:
                            factors.append((label, desc))
                    elif feat == "amount":
                        if val >= thr:
                            factors.append((label, f"₹{val:,.0f} (above threshold)"))
                    elif feat == "amount_deviation_pct":
                        if val >= thr:
                            factors.append((label, f"{val:.0f}% deviation"))
                    elif feat == "transaction_velocity":
                        if val >= thr:
                            factors.append((label, f"{val} txns (elevated)"))
                    elif feat == "location_change_km":
                        if val >= thr:
                            factors.append((label, f"{val:.0f} km"))
                    elif feat == "failed_attempts":
                        if val >= thr:
                            factors.append((label, f"{val} attempts"))
                    elif feat == "beneficiary_fan_in":
                        if val >= thr:
                            factors.append((label, f"{val} (elevated)"))
                    elif feat == "reversed_attempts":
                        if val >= thr:
                            factors.append((label, f"{val} reversals"))
                    elif feat == "approval_delay_sec":
                        if val <= thr:
                            factors.append((label, f"{val:.1f}s (suspiciously fast)"))
        return factors
