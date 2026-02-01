"""
LLM-Based Fraud Detection using Groq API
Provides intelligent fraud detection using Large Language Models
"""

import os
import json
import time
from typing import Dict, List, Tuple
import pandas as pd
from groq import Groq
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
load_dotenv()

class LLMFraudDetector:
    """
    LLM-based fraud detection using Groq API
    Analyzes transaction patterns using natural language understanding
    """
    
    def __init__(self, api_key: str = None, model: str = "llama-3.3-70b-versatile"):
        """
        Initialize LLM fraud detector
        
        Args:
            api_key: Groq API key (if None, loads from environment)
            model: Groq model to use
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError("GROQ_API_KEY not found. Set it in .env file or pass as parameter")
        
        self.model = model
        self.client = Groq(api_key=self.api_key)
        self.predictions = []
        self.processing_times = []
        
    def _create_fraud_detection_prompt(self, transaction: Dict) -> str:
        """
        Create a structured prompt for fraud detection
        
        Args:
            transaction: Dictionary containing transaction details
        
        Returns:
            Formatted prompt string
        """
        
        prompt = f"""You are an expert fraud detection system analyzing UPI (Unified Payments Interface) transactions in real-time.

Analyze the following transaction and determine if it is FRAUDULENT or LEGITIMATE.

TRANSACTION DETAILS:
- Amount: ₹{transaction.get('amount', 0):.2f}
- Time: {transaction.get('hour', 'N/A')}:00 hours
- Day of Week: {transaction.get('day_of_week', 'N/A')} (0=Monday, 6=Sunday)
- Is Weekend: {transaction.get('is_weekend', 'N/A')}
- Is Night Time: {transaction.get('is_night', 'N/A')}
- Transaction Velocity: {transaction.get('transaction_velocity', 0)} transactions/hour
- Failed Attempts: {transaction.get('failed_attempts', 0)}
- Device Changed: {transaction.get('device_change', 'N/A')}
- Location Change: {transaction.get('location_change_km', 0):.1f} km from last transaction
- Is High Amount: {transaction.get('is_high_amount', 'N/A')}
- Is Round Amount: {transaction.get('is_round_amount', 'N/A')}
- Is High Velocity: {transaction.get('is_high_velocity', 'N/A')}
- Is Location Jump: {transaction.get('is_location_jump', 'N/A')}
- Risk Score: {transaction.get('risk_score', 0)}

FRAUD INDICATORS TO CONSIDER:
1. Unusual transaction timing (late night/early morning)
2. High transaction velocity (multiple transactions quickly)
3. Large location changes in short time
4. Failed authentication attempts
5. Device changes
6. Abnormally high amounts
7. Suspicious behavioral patterns

INSTRUCTIONS:
Respond ONLY with a valid JSON object in this exact format:
{{
    "prediction": "FRAUD" or "LEGITIMATE",
    "confidence": <float between 0 and 1>,
    "reasoning": "<brief explanation>",
    "risk_factors": ["<factor1>", "<factor2>", ...]
}}

Do not include any text before or after the JSON object."""

        return prompt
    
    def predict_single(self, transaction: Dict) -> Tuple[int, float, str, List[str]]:
        """
        Predict if a single transaction is fraudulent using LLM
        
        Args:
            transaction: Transaction data as dictionary
        
        Returns:
            Tuple of (prediction, confidence, reasoning, risk_factors)
            prediction: 1 for fraud, 0 for legitimate
        """
        
        start_time = time.time()
        
        try:
            prompt = self._create_fraud_detection_prompt(transaction)
            
            # Call Groq API
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert fraud detection AI. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent results
                max_tokens=500
            )
            
            # Parse response
            response_text = response.choices[0].message.content.strip()
            
            # Try to extract JSON if there's extra text
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0].strip()
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0].strip()
            
            result = json.loads(response_text)
            
            # Extract values
            prediction = 1 if result['prediction'].upper() == 'FRAUD' else 0
            confidence = float(result.get('confidence', 0.5))
            reasoning = result.get('reasoning', 'No reasoning provided')
            risk_factors = result.get('risk_factors', [])
            
            processing_time = time.time() - start_time
            self.processing_times.append(processing_time)
            
            return prediction, confidence, reasoning, risk_factors
            
        except Exception as e:
            print(f"Error in LLM prediction: {e}")
            # Default to legitimate with low confidence on error
            return 0, 0.5, f"Error: {str(e)}", []
    
    def predict_batch(self, transactions_df: pd.DataFrame, max_samples: int = 100) -> pd.DataFrame:
        """
        Predict fraud for a batch of transactions
        
        Args:
            transactions_df: DataFrame of transactions
            max_samples: Maximum number of samples to process (to manage API costs)
        
        Returns:
            DataFrame with predictions and analysis
        """
        
        print(f"\n{'='*70}")
        print(f"LLM-BASED FRAUD DETECTION (Groq API)")
        print(f"{'='*70}")
        print(f"Model: {self.model}")
        print(f"Processing {min(len(transactions_df), max_samples)} transactions...")
        print(f"{'='*70}\n")
        
        # Sample if needed
        if len(transactions_df) > max_samples:
            df_sample = transactions_df.sample(n=max_samples, random_state=42)
            print(f"Note: Sampling {max_samples} transactions due to API cost limits")
        else:
            df_sample = transactions_df.copy()
        
        results = []
        
        for idx, row in df_sample.iterrows():
            # Convert row to dictionary
            transaction = row.to_dict()
            
            # Get prediction
            pred, conf, reasoning, risk_factors = self.predict_single(transaction)
            
            results.append({
                'transaction_idx': idx,
                'true_label': row.get('is_fraud', None),
                'llm_prediction': pred,
                'llm_confidence': conf,
                'llm_reasoning': reasoning,
                'llm_risk_factors': risk_factors
            })
            
            # Print progress
            if (len(results)) % 10 == 0:
                print(f"Processed {len(results)}/{len(df_sample)} transactions...")
            
            # Rate limiting (avoid hitting API limits)
            time.sleep(0.1)
        
        results_df = pd.DataFrame(results)
        
        # Calculate metrics if true labels available
        if 'is_fraud' in df_sample.columns:
            from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
            
            accuracy = accuracy_score(results_df['true_label'], results_df['llm_prediction'])
            precision = precision_score(results_df['true_label'], results_df['llm_prediction'], zero_division=0)
            recall = recall_score(results_df['true_label'], results_df['llm_prediction'], zero_division=0)
            f1 = f1_score(results_df['true_label'], results_df['llm_prediction'], zero_division=0)
            
            print(f"\n{'='*70}")
            print(f"LLM FRAUD DETECTION RESULTS")
            print(f"{'='*70}")
            print(f"Accuracy:  {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall:    {recall:.4f}")
            print(f"F1-Score:  {f1:.4f}")
            print(f"Avg Processing Time: {sum(self.processing_times)/len(self.processing_times):.3f}s per transaction")
            print(f"{'='*70}\n")
            
            # Store metrics
            self.metrics = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1,
                'avg_processing_time': sum(self.processing_times)/len(self.processing_times)
            }
        
        return results_df
    
    def analyze_sample_predictions(self, results_df: pd.DataFrame, n_samples: int = 5):
        """
        Display sample predictions with reasoning
        
        Args:
            results_df: Results dataframe from predict_batch
            n_samples: Number of samples to display
        """
        
        print(f"\n{'='*70}")
        print(f"SAMPLE LLM PREDICTIONS WITH REASONING")
        print(f"{'='*70}\n")
        
        # Show fraud cases
        fraud_samples = results_df[results_df['llm_prediction'] == 1].head(n_samples)
        
        print("FRAUD PREDICTIONS:")
        print("-" * 70)
        for idx, row in fraud_samples.iterrows():
            print(f"\nTransaction #{row['transaction_idx']}")
            print(f"  True Label: {'FRAUD' if row['true_label'] == 1 else 'LEGITIMATE'}")
            print(f"  LLM Prediction: FRAUD")
            print(f"  Confidence: {row['llm_confidence']:.2%}")
            print(f"  Reasoning: {row['llm_reasoning']}")
            print(f"  Risk Factors: {', '.join(row['llm_risk_factors'])}")
        
        # Show legitimate cases
        print("\n" + "="*70)
        print("LEGITIMATE PREDICTIONS:")
        print("-" * 70)
        legit_samples = results_df[results_df['llm_prediction'] == 0].head(n_samples)
        
        for idx, row in legit_samples.iterrows():
            print(f"\nTransaction #{row['transaction_idx']}")
            print(f"  True Label: {'FRAUD' if row['true_label'] == 1 else 'LEGITIMATE'}")
            print(f"  LLM Prediction: LEGITIMATE")
            print(f"  Confidence: {row['llm_confidence']:.2%}")
            print(f"  Reasoning: {row['llm_reasoning']}")
        
        print("\n" + "="*70)


class HybridFraudDetector:
    """
    Hybrid fraud detection combining ML models and LLM
    """
    
    def __init__(self, ml_model, llm_detector: LLMFraudDetector):
        """
        Initialize hybrid detector
        
        Args:
            ml_model: Trained ML model
            llm_detector: LLM fraud detector instance
        """
        self.ml_model = ml_model
        self.llm_detector = llm_detector
    
    def predict_hybrid(self, transaction_features, transaction_dict, 
                       strategy='weighted', ml_weight=0.6, llm_weight=0.4):
        """
        Make hybrid prediction combining ML and LLM
        
        Args:
            transaction_features: Preprocessed features for ML model
            transaction_dict: Raw transaction dict for LLM
            strategy: 'weighted', 'consensus', or 'llm_verify'
            ml_weight: Weight for ML prediction (if weighted strategy)
            llm_weight: Weight for LLM prediction (if weighted strategy)
        
        Returns:
            Final prediction and explanation
        """
        
        # Get ML prediction
        ml_pred = self.ml_model.predict([transaction_features])[0]
        ml_proba = self.ml_model.predict_proba([transaction_features])[0][1] if hasattr(self.ml_model, 'predict_proba') else ml_pred
        
        # Get LLM prediction
        llm_pred, llm_conf, reasoning, risk_factors = self.llm_detector.predict_single(transaction_dict)
        
        if strategy == 'weighted':
            # Weighted average of probabilities
            final_score = (ml_weight * ml_proba) + (llm_weight * llm_conf * llm_pred)
            final_pred = 1 if final_score > 0.5 else 0
            
        elif strategy == 'consensus':
            # Both must agree for fraud
            final_pred = 1 if (ml_pred == 1 and llm_pred == 1) else 0
            
        elif strategy == 'llm_verify':
            # ML flags, LLM verifies
            if ml_pred == 0:
                final_pred = 0
            else:
                final_pred = llm_pred
        else:
            final_pred = ml_pred
        
        return {
            'final_prediction': final_pred,
            'ml_prediction': ml_pred,
            'ml_confidence': ml_proba,
            'llm_prediction': llm_pred,
            'llm_confidence': llm_conf,
            'llm_reasoning': reasoning,
            'llm_risk_factors': risk_factors
        }


if __name__ == "__main__":
    print("LLM Fraud Detector Module")
    print("Set GROQ_API_KEY in .env file to use this module")
    print("\nExample usage:")
    print("  from llm_detector import LLMFraudDetector")
    print("  detector = LLMFraudDetector()")
    print("  results = detector.predict_batch(transactions_df)")
