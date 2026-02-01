"""
LLM-Based Fraud Detection using Groq API
Provides intelligent fraud detection using large language models
"""

import os
import json
import pandas as pd
import numpy as np
from groq import Groq
from dotenv import load_dotenv
from tqdm import tqdm
import time

# Load environment variables
load_dotenv()

class GroqFraudDetector:
    """
    LLM-based fraud detection using Groq API
    Analyzes transaction context and patterns using natural language understanding
    """
    
    def __init__(self, api_key=None, model="llama-3.3-70b-versatile"):
        """
        Initialize Groq fraud detector
        
        Args:
            api_key: Groq API key (if None, loads from GROQ_API_KEY env variable)
            model: Groq model to use
        """
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Groq API key required. Set GROQ_API_KEY environment variable or pass api_key parameter.\n"
                "Get your API key from: https://console.groq.com/keys"
            )
        
        self.client = Groq(api_key=self.api_key)
        self.model = model
        self.performance_metrics = {}
        
    def create_fraud_detection_prompt(self, transaction):
        """
        Create a detailed prompt for fraud detection
        
        Args:
            transaction: Dictionary containing transaction details
        
        Returns:
            Formatted prompt string
        """
        
        prompt = f"""You are an expert fraud detection system for UPI (Unified Payments Interface) transactions in India.

Analyze the following transaction and determine if it is FRAUDULENT or LEGITIMATE.

TRANSACTION DETAILS:
- Amount: ₹{transaction.get('amount', 0):.2f}
- Hour of Day: {transaction.get('hour', 'Unknown')} (24-hour format)
- Day of Week: {transaction.get('day_of_week', 'Unknown')} (0=Monday, 6=Sunday)
- Is Weekend: {'Yes' if transaction.get('is_weekend', 0) == 1 else 'No'}
- Transaction Velocity: {transaction.get('transaction_velocity', 0)} transactions/hour
- Failed Attempts: {transaction.get('failed_attempts', 0)}
- Device Changed: {'Yes' if transaction.get('device_change', 0) == 1 else 'No'}
- Location Change: {transaction.get('location_change_km', 0):.2f} km from last transaction
- Is Night Time: {'Yes' if transaction.get('is_night', 0) == 1 else 'No'}

FRAUD INDICATORS TO CONSIDER:
1. Unusual transaction amounts (very high or suspiciously round numbers)
2. Odd hours (late night/early morning transactions)
3. High transaction velocity (multiple transactions in short time)
4. Multiple failed attempts before success
5. Sudden device changes
6. Large geographic location jumps
7. Weekend activity patterns
8. High-risk combinations of above factors

RESPONSE FORMAT (JSON only):
{{
    "prediction": "FRAUD" or "LEGITIMATE",
    "confidence": <float between 0 and 1>,
    "reasoning": "<brief explanation of decision>",
    "risk_factors": [<list of identified risk factors>]
}}

Respond ONLY with valid JSON, no additional text."""

        return prompt
    
    def predict_single(self, transaction):
        """
        Predict fraud for a single transaction
        
        Args:
            transaction: Dictionary containing transaction details
        
        Returns:
            Dictionary with prediction, confidence, and reasoning
        """
        
        prompt = self.create_fraud_detection_prompt(transaction)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a fraud detection expert. Always respond with valid JSON only."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,  # Low temperature for consistent predictions
                max_tokens=500
            )
            
            # Parse response
            content = response.choices[0].message.content.strip()
            
            # Extract JSON if wrapped in markdown
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            
            result = json.loads(content)
            
            return {
                'prediction': 1 if result['prediction'].upper() == 'FRAUD' else 0,
                'confidence': result.get('confidence', 0.5),
                'reasoning': result.get('reasoning', ''),
                'risk_factors': result.get('risk_factors', [])
            }
            
        except Exception as e:
            print(f"Error in LLM prediction: {e}")
            # Return neutral prediction on error
            return {
                'prediction': 0,
                'confidence': 0.5,
                'reasoning': f'Error: {str(e)}',
                'risk_factors': []
            }
    
    def predict_batch(self, transactions_df, max_samples=100, delay=0.5):
        """
        Predict fraud for a batch of transactions
        
        Args:
            transactions_df: DataFrame with transaction data
            max_samples: Maximum number of samples to process (to manage API costs)
            delay: Delay between API calls (seconds) to avoid rate limits
        
        Returns:
            DataFrame with predictions
        """
        
        print(f"\n🤖 Running LLM-Based Fraud Detection (Groq API)")
        print(f"   Model: {self.model}")
        print(f"   Processing {min(len(transactions_df), max_samples)} transactions...")
        
        # Sample if needed
        if len(transactions_df) > max_samples:
            # Stratified sampling to maintain fraud ratio
            fraud_samples = transactions_df[transactions_df['is_fraud'] == 1].sample(
                n=min(len(transactions_df[transactions_df['is_fraud'] == 1]), max_samples // 2),
                random_state=42
            )
            legit_samples = transactions_df[transactions_df['is_fraud'] == 0].sample(
                n=max_samples - len(fraud_samples),
                random_state=42
            )
            sample_df = pd.concat([fraud_samples, legit_samples]).sample(frac=1, random_state=42)
        else:
            sample_df = transactions_df.copy()
        
        predictions = []
        confidences = []
        reasonings = []
        
        for idx, row in tqdm(sample_df.iterrows(), total=len(sample_df), desc="Processing"):
            transaction = row.to_dict()
            result = self.predict_single(transaction)
            
            predictions.append(result['prediction'])
            confidences.append(result['confidence'])
            reasonings.append(result['reasoning'])
            
            # Rate limiting
            time.sleep(delay)
        
        # Add predictions to dataframe
        result_df = sample_df.copy()
        result_df['llm_prediction'] = predictions
        result_df['llm_confidence'] = confidences
        result_df['llm_reasoning'] = reasonings
        
        print(f"✓ LLM predictions completed!")
        
        return result_df
    
    def evaluate(self, results_df):
        """
        Evaluate LLM performance
        
        Args:
            results_df: DataFrame with actual labels and LLM predictions
        
        Returns:
            Dictionary of performance metrics
        """
        
        from sklearn.metrics import (
            accuracy_score, precision_score, recall_score, 
            f1_score, confusion_matrix, roc_auc_score
        )
        
        y_true = results_df['is_fraud'].values
        y_pred = results_df['llm_prediction'].values
        y_conf = results_df['llm_confidence'].values
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_conf) if len(np.unique(y_true)) > 1 else 0,
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'avg_confidence': y_conf.mean()
        }
        
        self.performance_metrics = metrics
        
        return metrics
    
    def print_evaluation(self, results_df):
        """Print detailed evaluation results"""
        
        metrics = self.evaluate(results_df)
        
        print(f"\n{'='*60}")
        print(f"LLM-BASED FRAUD DETECTION (Groq API)")
        print(f"{'='*60}")
        print(f"Model: {self.model}")
        print(f"Samples Analyzed: {len(results_df)}")
        print(f"\nPerformance Metrics:")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  Avg Confidence: {metrics['avg_confidence']:.4f}")
        print(f"\nConfusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"{'='*60}\n")
        
        return metrics
    
    def show_sample_predictions(self, results_df, n=5):
        """Show sample predictions with reasoning"""
        
        print(f"\n{'='*80}")
        print(f"SAMPLE LLM PREDICTIONS WITH REASONING")
        print(f"{'='*80}\n")
        
        # Show mix of correct and incorrect predictions
        correct = results_df[results_df['is_fraud'] == results_df['llm_prediction']].head(n//2)
        incorrect = results_df[results_df['is_fraud'] != results_df['llm_prediction']].head(n//2)
        samples = pd.concat([correct, incorrect])
        
        for idx, row in samples.iterrows():
            actual = "FRAUD" if row['is_fraud'] == 1 else "LEGITIMATE"
            predicted = "FRAUD" if row['llm_prediction'] == 1 else "LEGITIMATE"
            correct_pred = "✓" if actual == predicted else "✗"
            
            print(f"{correct_pred} Transaction #{idx}")
            print(f"  Amount: ₹{row['amount']:.2f} | Hour: {row['hour']} | Velocity: {row['transaction_velocity']}")
            print(f"  Actual: {actual} | Predicted: {predicted} | Confidence: {row['llm_confidence']:.2f}")
            print(f"  Reasoning: {row['llm_reasoning']}")
            print("-" * 80)
        
        print()


def create_env_template():
    """Create .env template file for API key"""
    
    template = """# Groq API Configuration
# Get your API key from: https://console.groq.com/keys

GROQ_API_KEY=your_groq_api_key_here
"""
    
    with open('../.env.template', 'w') as f:
        f.write(template)
    
    print("Created .env.template file")
    print("Copy it to .env and add your Groq API key")


if __name__ == "__main__":
    print("Testing Groq Fraud Detector...")
    print("\nFirst, set up your API key:")
    print("1. Get API key from: https://console.groq.com/keys")
    print("2. Create .env file with: GROQ_API_KEY=your_key_here")
    print("3. Or set environment variable: export GROQ_API_KEY=your_key_here")
    
    create_env_template()
    
    # Test with sample transaction
    sample_transaction = {
        'amount': 15000.0,
        'hour': 2,
        'day_of_week': 6,
        'is_weekend': 1,
        'transaction_velocity': 8,
        'failed_attempts': 2,
        'device_change': 1,
        'location_change_km': 150.0,
        'is_night': 1
    }
    
    print("\nSample transaction (likely fraud):")
    print(json.dumps(sample_transaction, indent=2))
    
    try:
        detector = GroqFraudDetector()
        result = detector.predict_single(sample_transaction)
        print("\nLLM Prediction:")
        print(json.dumps(result, indent=2))
    except ValueError as e:
        print(f"\n{e}")
