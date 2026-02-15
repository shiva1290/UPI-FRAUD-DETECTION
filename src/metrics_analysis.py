"""
Metrics analysis content (SRP).
Provides explanatory text for precision, recall, FP/FN, and ML vs ML+LLM comparison.
"""


def get_metrics_guide() -> dict:
    """
    Returns structured content for the dashboard metrics guide.
    OCP: extend by adding new keys without modifying callers.
    """
    return {
        "title": "Why These Metrics Matter",
        "recall_importance": {
            "heading": "Why Recall Matters in Fraud Detection",
            "content": (
                "Recall (True Positives / (True Positives + False Negatives)) measures "
                "what fraction of actual frauds we catch. In fraud detection, missing a "
                "fraudulent transaction (false negative) is often costlier than flagging "
                "a legitimate one (false positive). Low recall means fraud slips through; "
                "high recall reduces financial loss and customer trust issues. We prioritize "
                "recall alongside precision to balance catching fraud vs. minimizing false alarms."
            ),
        },
        "false_positives_vs_false_negatives": {
            "heading": "False Positives vs False Negatives",
            "content": (
                "False Positive: Legitimate transaction wrongly flagged as fraud. "
                "Consequences: customer friction, blocked payments, support load. "
                "False Negative: Fraudulent transaction wrongly allowed. "
                "Consequences: direct financial loss, chargebacks, regulatory risk. "
                "For UPI fraud, false negatives are typically more costly. We use "
                "F1-score to balance precision and recall, and ROC-AUC to assess "
                "ranking quality across thresholds."
            ),
        },
        "metrics_definitions": {
            "precision": "Of transactions flagged as fraud, how many were actually fraud?",
            "recall": "Of all actual frauds, how many did we catch?",
            "f1_score": "Harmonic mean of precision and recall; balances both.",
        },
    }


def get_ml_vs_llm_comparison() -> dict:
    """Returns qualitative comparison of ML-only vs ML+LLM system behavior."""
    return {
        "title": "ML-Only vs ML+LLM: Qualitative Comparison",
        "ml_only": {
            "heading": "ML-Only (e.g., Random Forest)",
            "strengths": [
                "Fast inference (~30ms per transaction)",
                "Consistent, deterministic behavior",
                "No external API dependency",
                "Scales to high throughput",
            ],
            "limitations": [
                "No human-readable reasoning",
                "Black-box; harder to explain to auditors",
                "Fixed feature set; limited context awareness",
            ],
        },
        "ml_plus_llm": {
            "heading": "ML + LLM (Hybrid)",
            "strengths": [
                "Natural-language explanations for flagged transactions",
                "Aligns with model features (amount, velocity, device change)",
                "Useful for compliance and dispute resolution",
                "LLM invoked only for medium/high risk (cost-efficient)",
            ],
            "limitations": [
                "Slower for LLM path (~2–5s per call)",
                "Requires Groq API key and incurs API cost",
                "Non-deterministic; may vary across runs",
            ],
        },
        "recommendation": (
            "Use ML for real-time screening; use LLM for explaining and reviewing "
            "suspicious cases. The risk-based decision layer invokes LLM only for "
            "medium and high risk, keeping most transactions fast and cost-effective."
        ),
    }
