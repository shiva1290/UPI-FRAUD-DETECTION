"""
Demo script to test LLM-based fraud detection
Tests a few sample transactions with Groq API
"""

import os
from dotenv import load_dotenv
from llm_detector import LLMFraudDetector
import pandas as pd

# Load environment variables
load_dotenv()

def test_llm_detection():
    """Test LLM fraud detection with sample transactions"""
    
    print("="*70)
    print(" LLM FRAUD DETECTION - DEMO")
    print("="*70)
    
    # Check if API key is set
    if not os.getenv("GROQ_API_KEY"):
        print("\n❌ ERROR: GROQ_API_KEY not found!")
        print("\nTo use LLM detection:")
        print("  1. Get your API key from https://console.groq.com/keys")
        print("  2. Copy .env.example to .env")
        print("  3. Add your API key to .env file")
        print("\nExample .env file:")
        print("  GROQ_API_KEY=gsk_your_actual_api_key_here")
        return
    
    # Initialize detector
    try:
        detector = LLMFraudDetector()
        print(f"\n✓ LLM Detector initialized")
        print(f"  Model: {detector.model}")
    except Exception as e:
        print(f"\n❌ Error initializing detector: {e}")
        return
    
    # Sample transactions
    print("\n" + "="*70)
    print(" TESTING SAMPLE TRANSACTIONS")
    print("="*70)
    
    transactions = [
        {
            "name": "Suspicious Night Transaction",
            "data": {
                "amount": 8500,
                "hour": 2,
                "day_of_week": 3,
                "is_weekend": 0,
                "is_night": 1,
                "transaction_velocity": 9,
                "failed_attempts": 3,
                "device_change": 1,
                "location_change_km": 250,
                "is_high_amount": 1,
                "is_round_amount": 0,
                "is_high_velocity": 1,
                "is_location_jump": 1,
                "risk_score": 27
            }
        },
        {
            "name": "Normal Daytime Purchase",
            "data": {
                "amount": 450,
                "hour": 14,
                "day_of_week": 2,
                "is_weekend": 0,
                "is_night": 0,
                "transaction_velocity": 2,
                "failed_attempts": 0,
                "device_change": 0,
                "location_change_km": 2.5,
                "is_high_amount": 0,
                "is_round_amount": 0,
                "is_high_velocity": 0,
                "is_location_jump": 0,
                "risk_score": 0
            }
        },
        {
            "name": "High Amount Weekend",
            "data": {
                "amount": 15000,
                "hour": 11,
                "day_of_week": 6,
                "is_weekend": 1,
                "is_night": 0,
                "transaction_velocity": 1,
                "failed_attempts": 0,
                "device_change": 0,
                "location_change_km": 5,
                "is_high_amount": 1,
                "is_round_amount": 1,
                "is_high_velocity": 0,
                "is_location_jump": 0,
                "risk_score": 0
            }
        }
    ]
    
    # Test each transaction
    for i, txn in enumerate(transactions, 1):
        print(f"\n{'─'*70}")
        print(f"Transaction {i}: {txn['name']}")
        print(f"{'─'*70}")
        print(f"Amount: ₹{txn['data']['amount']}")
        print(f"Time: {txn['data']['hour']}:00")
        print(f"Velocity: {txn['data']['transaction_velocity']} txns/hour")
        print(f"Location change: {txn['data']['location_change_km']} km")
        print(f"Failed attempts: {txn['data']['failed_attempts']}")
        
        # Get prediction
        pred, conf, reasoning, risk_factors = detector.predict_single(txn['data'])
        
        print(f"\n🤖 LLM Analysis:")
        print(f"  Prediction: {'🚨 FRAUD' if pred == 1 else '✅ LEGITIMATE'}")
        print(f"  Confidence: {conf:.1%}")
        print(f"  Reasoning: {reasoning}")
        if risk_factors:
            print(f"  Risk Factors:")
            for factor in risk_factors:
                print(f"    • {factor}")
    
    print("\n" + "="*70)
    print(" DEMO COMPLETED")
    print("="*70)
    print("\nℹ️  This demonstrates how LLM provides:")
    print("  ✓ Binary prediction (fraud/legitimate)")
    print("  ✓ Confidence score")
    print("  ✓ Human-readable reasoning")
    print("  ✓ Specific risk factors identified")
    print("\n💡 Use this in your paper to show explainable AI!")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_llm_detection()
