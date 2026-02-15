"""
LLM-Based Fraud Detector. Uses LLMClient abstraction (DIP).
"""

import os
import json
import time
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from llm_client import LLMClient, GroqLLMClient


def _clean_api_key(key):
    """Strip quotes, whitespace, and line endings from API key."""
    if not key:
        return key
    return key.strip().strip('"').strip("'").replace('\r', '').replace('\n', '')


class LLMFraudDetector:
    """LLM fraud detector. Depends on LLMClient abstraction (DIP)."""

    def __init__(self, api_key=None, model=None, client: LLMClient = None):
        raw = api_key or os.environ.get('GROQ_API_KEY') or ''
        self.api_key = _clean_api_key(raw)
        self.model = model or os.environ.get('LLM_MODEL', 'llama-3.3-70b-versatile')

        if client is not None:
            self._llm_client = client
        else:
            if not self.api_key:
                raise ValueError("GROQ_API_KEY not found. Please set it in .env file.")
            self._llm_client = GroqLLMClient(api_key=self.api_key, model=self.model)

        self.metrics = {}
        
    def _create_prompt(self, transaction):
        """Create a prompt for the LLM based on transaction data"""
        # Convert pandas series to dict if needed
        if hasattr(transaction, 'to_dict'):
            data = transaction.to_dict()
        else:
            data = transaction
            
        # Select relevant fields to reduce token usage
        fields = [
            'amount', 'hour', 'is_night', 'is_weekend', 
            'transaction_velocity', 'location_change_km',
            'device_change', 'failed_attempts', 'beneficiary_fan_in'
        ]
        
        context = {k: data.get(k, 'N/A') for k in fields}
        
        prompt = f"""
        Analyze this UPI transaction for fraud risk.
        
        Transaction Data:
        {json.dumps(context, indent=2)}
        
        Task:
        Determine if this transaction is fraudulent (is_fraud=1) or legitimate (is_fraud=0).
        Provide a confidence score (0-100), reasoning, and list of risk factors.
        
        Output JSON format only:
        {{
            "is_fraud": boolean,
            "confidence": number,
            "reasoning": "string",
            "risk_factors": ["string", "string"]
        }}
        """
        return prompt

    def predict_single(self, transaction_data):
        """
        Predict fraud for a single transaction
        Returns: prediction (0/1), confidence, reasoning, risk_factors
        """
        try:
            prompt = self._create_prompt(transaction_data)
            messages = [
                {"role": "system", "content": "You are an expert financial fraud detection system. Output valid JSON only."},
                {"role": "user", "content": prompt}
            ]
            response_text = self._llm_client.complete(messages, temperature=0.1)
            result = json.loads(response_text)
            
            prediction = 1 if result.get('is_fraud', False) else 0
            confidence = result.get('confidence', 0)
            reasoning = result.get('reasoning', "No reasoning provided")
            risk_factors = result.get('risk_factors', [])
            
            return prediction, confidence, reasoning, risk_factors
            
        except Exception as e:
            print(f"LLM Prediction Error: {e}")
            # Fallback safe return
            return 0, 0.0, f"Error: {str(e)}", []

    def predict_batch(self, df, max_samples=100):
        """
        Run predictions on a batch of transactions (DataFrame)
        """
        results = []
        actuals = []
        predictions = []
        
        # Limit samples
        samples = df.head(max_samples).copy()
        print(f"   Processing {len(samples)} transactions with {self.model}...")
        
        for i, (index, row) in enumerate(samples.iterrows()):
            # Rate limiting (simple)
            if i > 0 and i % 10 == 0:
                time.sleep(1)
                print(f"   Processed {i}/{len(samples)}...")
                
            pred, conf, reason, risks = self.predict_single(row)
            
            results.append({
                'transaction_id': row.get('transaction_id', index),
                'actual_is_fraud': row.get('is_fraud', 0),
                'llm_prediction': pred,
                'llm_confidence': conf,
                'llm_reasoning': reason,
                'risk_factors': str(risks)
            })
            
            actuals.append(row.get('is_fraud', 0))
            predictions.append(pred)
            
        # Calculate metrics
        if actuals and predictions:
            self.metrics = {
                'accuracy': accuracy_score(actuals, predictions),
                'precision': precision_score(actuals, predictions, zero_division=0),
                'recall': recall_score(actuals, predictions, zero_division=0),
                'f1_score': f1_score(actuals, predictions, zero_division=0)
            }
            
        return pd.DataFrame(results)

    def analyze_sample_predictions(self, results_df, n_samples=3):
        """Print analysis of sample predictions"""
        print("\n🔍 LLM Analysis Samples:")
        print("-" * 60)
        # Column is 'llm_prediction' (set by predict_batch), not 'llm_is_fraud'
        pred_col = 'llm_prediction' if 'llm_prediction' in results_df.columns else None
        if pred_col is None:
            print("  No prediction column found (LLM may have failed for all rows).")
            return
        frauds = results_df[results_df[pred_col] == 1]
        samples = frauds.head(n_samples) if not frauds.empty else results_df.head(n_samples)
        for _, row in samples.iterrows():
            status = "✅ CORRECT" if row[pred_col] == row['actual_is_fraud'] else "❌ INCORRECT"
            print(f"Transaction ID: {row['transaction_id']}")
            print(f"Prediction: {'FRAUD' if row[pred_col] else 'LEGIT'} ({status})")
            print(f"Confidence: {row['llm_confidence']}%")
            print(f"Reasoning: {row['llm_reasoning']}")
            print("-" * 60)