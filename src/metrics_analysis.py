"""
Metrics analysis content (SRP).
Provides explanatory text for precision, recall, FP/FN, imbalanced data,
and ML classifiers vs LLM explanation module.
"""


def get_metrics_guide() -> dict:
    """
    Returns structured content for the dashboard metrics guide.
    OCP: extend by adding new keys without modifying callers.
    """
    return {
        "title": "Why These Metrics Matter",
        "imbalanced_dataset": {
            "heading": "Fraud Datasets Are Imbalanced",
            "content": (
                "Fraud detection datasets typically have far fewer fraudulent transactions "
                "(often 1–5%) than legitimate ones. In such imbalanced settings, accuracy is misleading: "
                "a model that predicts 'legitimate' for everything can achieve 95%+ accuracy while "
                "missing all fraud. We prioritize Precision, Recall, and F1-score instead of accuracy."
            ),
        },
        "recall_importance": {
            "heading": "Why Recall Matters in Fraud Detection",
            "content": (
                "Recall (True Positives / (True Positives + False Negatives)) measures "
                "what fraction of actual frauds we catch. In fraud detection, missing a "
                "fraudulent transaction (false negative) is often costlier than flagging "
                "a legitimate one (false positive). Low recall means fraud slips through; "
                "high recall reduces financial loss and customer trust issues. We balance "
                "recall with precision to avoid too many false alarms."
            ),
        },
        "false_positives_vs_false_negatives": {
            "heading": "False Positives vs False Negatives: Impact",
            "content": (
                "False Positive: A legitimate transaction wrongly flagged as fraud. "
                "Impact: customer friction, blocked payments, support load, lost revenue. "
                "False Negative: A fraudulent transaction wrongly allowed. "
                "Impact: direct financial loss, chargebacks, regulatory risk, reputational damage. "
                "For UPI fraud, false negatives are typically more costly. We use F1-score "
                "to balance both, and ROC-AUC to assess ranking quality across thresholds."
            ),
        },
        "selected_model": {
            "heading": "Random Forest: Selected Model",
            "content": (
                "Random Forest is the final selected model based on balanced Precision and Recall. "
                "It provides good F1-score and ROC-AUC, handles imbalanced data well, and supports "
                "feature importance for explainability. The LLM module adds human-readable reasoning "
                "for medium/high-risk cases; it is not a classifier."
            ),
        },
        "moderate_recall": {
            "heading": "Why Moderate Recall (~48%) Is Acceptable",
            "content": (
                "Moderate recall reduces false alarms: chasing very high recall often means "
                "lowering the decision threshold, which sharply increases false positives. "
                "Too many false alarms overload support, frustrate customers, and block legitimate "
                "payments. A ~48% recall balances catching fraud with manageable alert volume."
            ),
        },
        "threshold_tradeoff": (
            "Raising the decision threshold increases precision and lowers recall; lowering it does the opposite."
        ),
        "metrics_definitions": {
            "precision": "Of transactions flagged as fraud, how many were actually fraud?",
            "recall": "Of all actual frauds, how many did we catch? Moderate recall (~48%) limits false alarms.",
            "f1_score": "Harmonic mean of precision and recall; balances both.",
            "pr_auc": (
                "Area under the Precision–Recall curve (average precision). "
                "Focuses on performance for the positive (fraud) class and is more informative than ROC-AUC on imbalanced data."
            ),
            "roc_auc": (
                "Model's discrimination ability across thresholds—how well it ranks fraud vs legitimate—not accuracy. "
                "On highly imbalanced fraud data, ROC-AUC can look high even when precision on frauds is poor; PR-AUC is more sensitive."
            ),
        },
    }


def get_ml_vs_llm_comparison() -> dict:
    """
    Returns qualitative comparison of ML classifiers vs LLM explanation module.
    LLM is presented as an explanation module, not a prediction model.
    """
    return {
        "title": "ML Classifiers vs LLM Explanation Module",
        "ml_only": {
            "heading": "ML Classifiers (e.g., Random Forest)",
            "strengths": [
                "Fast inference (~30ms per transaction)",
                "Consistent, deterministic behavior",
                "No external API dependency",
                "Scales to high throughput",
                "Classifies fraud vs legitimate; produces risk scores",
            ],
            "limitations": [
                "No human-readable reasoning",
                "Black-box; harder to explain to auditors",
                "Fixed feature set; limited context awareness",
            ],
        },
        "llm_explanation": {
            "heading": "LLM Explanation Module (Groq)",
            "strengths": [
                "Natural-language explanations for flagged transactions",
                "Aligns with model features (amount, velocity, device change)",
                "Useful for compliance and dispute resolution",
                "Invoked only for medium/high risk (cost-efficient)",
                "Does not replace ML; adds interpretability",
            ],
            "limitations": [
                "Slower (~2–5s per call)",
                "Requires Groq API key and incurs API cost",
                "Non-deterministic; may vary across runs",
            ],
        },
        "recommendation": (
            "Use ML classifiers for real-time fraud classification. Use the LLM explanation module "
            "to generate human-readable reasoning for medium/high-risk cases. The LLM is not a "
            "classifier; it explains why a transaction was flagged."
        ),
    }
