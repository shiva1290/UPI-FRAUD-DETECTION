"""
LLM-based explanation module.
Uses LLMClient abstraction (DIP) to generate natural-language reasoning
from a risk score and top SHAP features. It does NOT act as a classifier.
"""

import os
import json
from typing import List, Dict, Any, Tuple, Optional

from llm_client import LLMClient, GroqLLMClient


def _clean_api_key(key: Optional[str]) -> Optional[str]:
    """Strip quotes, whitespace, and line endings from API key."""
    if not key:
        return key
    return key.strip().strip('"').strip("'").replace('\r', '').replace('\n', '')


class LLMFraudDetector:
    """
    LLM explanation helper.

    Given a risk_score and top SHAP features, it asks the LLM to
    produce reasoning and risk_factors. It does not return a fraud label
    or compute accuracy/precision/recall.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        client: Optional[LLMClient] = None,
        feature_names: Optional[List[str]] = None,
    ):
        raw = api_key or os.environ.get('GROQ_API_KEY') or ''
        self.api_key = _clean_api_key(raw)
        self.model = model or os.environ.get('LLM_MODEL', 'llama-3.3-70b-versatile')
        # feature_names is kept only for backwards compatibility; SHAP features are passed per-call
        self.feature_names = feature_names or []

        if client is not None:
            self._llm_client = client
        else:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not found. Please set it in .env file.")
            self._llm_client = GroqLLMClient(api_key=self.api_key, model=self.model)

    @staticmethod
    def _build_features_context(top_features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Normalise SHAP top features into a stable structure:
        [{name, value, shap_value}, ...], capped to the top 3–5 entries.
        """
        context: List[Dict[str, Any]] = []
        # Enforce a hard cap of 5 features; callers are expected to pass
        # pre-sorted SHAP features (highest |shap_value| first).
        limited = top_features[:5]
        for f in limited:
            context.append(
                {
                    "name": str(f.get("name")),
                    "value": float(f.get("value", 0.0)),
                    "shap_value": float(f.get("shap_value", 0.0)),
                }
            )
        return context

    def _create_prompt(self, risk_score: float, top_features: List[Dict[str, Any]]) -> str:
        """
        Create a prompt that ONLY exposes:
        - the ML risk score (0–100)
        - the top SHAP features (name, value, shap_value)
        """
        # Enforce 3–5 SHAP features in the prompt where possible.
        # Callers should already pass the top features (sorted by |SHAP|).
        if not top_features:
            features_for_prompt: List[Dict[str, Any]] = []
        else:
            # At most 5, at least as many as we actually have (up to 5)
            features_for_prompt = top_features[:5]

        features_context = self._build_features_context(features_for_prompt)

        prompt = f"""
You are an expert UPI fraud risk analyst.
You are given:
- A fraud risk score from a machine learning model (0–100, higher = more risky).
- The top contributing SHAP features from that model for this transaction.

Risk score: {risk_score:.1f}

Top SHAP features (feature, value, SHAP contribution for fraud class):
{json.dumps(features_context, indent=2)}

Instructions:
- Do NOT re-classify the transaction as fraud/legitimate.
- Treat the risk score as already computed by the ML model.
- Explain in clear, concise language WHY the risk score is at this level,
  referring explicitly to the listed features, their values, and whether they
  increase or decrease fraud risk.
- Use at most 120 words in the reasoning text.
- Summarise the main 2–5 risk factors as short bullet-style strings.

Output JSON only:
{{
  "reasoning": "string with natural language explanation (max 120 words, ideally 2–5 sentences)",
  "risk_factors": ["short bullet 1", "short bullet 2", "..."]
}}
"""
        return prompt

    def explain(
        self,
        risk_score: float,
        top_features: List[Dict[str, Any]],
        temperature: float = 0.1,
    ) -> Tuple[str, List[str]]:
        """
        Generate an LLM explanation given a risk score and top SHAP features.

        Returns:
            (reasoning, risk_factors)
        """
        try:
            prompt = self._create_prompt(risk_score, top_features)
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert financial fraud detection explainer. "
                               "Output valid JSON only, no extra text.",
                },
                {"role": "user", "content": prompt},
            ]
            response_text = self._llm_client.complete(messages, temperature=temperature)
            result = json.loads(response_text)

            reasoning = result.get("reasoning", "No reasoning provided")
            risk_factors = result.get("risk_factors", [])
            if not isinstance(risk_factors, list):
                risk_factors = []
            # Enforce a hard cap of ~120 words on the explanation.
            if isinstance(reasoning, str):
                words = reasoning.split()
                if len(words) > 120:
                    reasoning = " ".join(words[:120]) + " ..."
            # Cap risk factors list length to 5 entries.
            risk_factors = risk_factors[:5]
            return reasoning, risk_factors
        except Exception as e:
            # Fallback safe return; caller can still show something to the user
            return f"LLM explanation failed: {str(e)}", []
