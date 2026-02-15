"""
Comprehensive API Tests for UPI Fraud Detection System
Tests all endpoints with various scenarios
"""

import requests
import json
import time
from datetime import datetime

BASE_URL = "http://localhost:5000"

class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def print_test(name, status, message=""):
    """Print test result"""
    symbol = "✓" if status else "✗"
    color = Colors.GREEN if status else Colors.RED
    print(f"{color}{symbol}{Colors.END} {name}")
    if message:
        print(f"  {Colors.YELLOW}{message}{Colors.END}")

def test_health_check():
    """Test health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        success = response.status_code == 200 and response.json()['status'] == 'healthy'
        print_test("Health Check", success, f"Status: {response.status_code}")
        return success
    except Exception as e:
        print_test("Health Check", False, str(e))
        return False

def test_stats_endpoint():
    """Test statistics endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/stats")
        data = response.json()
        success = (
            response.status_code == 200 and
            'total_transactions' in data and
            'fraud_rate' in data
        )
        print_test("Statistics Endpoint", success, 
                  f"Total: {data.get('total_transactions', 'N/A')}, "
                  f"Fraud Rate: {data.get('fraud_rate', 'N/A')}%")
        return success
    except Exception as e:
        print_test("Statistics Endpoint", False, str(e))
        return False

def test_ml_prediction_valid():
    """Test ML prediction with valid data"""
    try:
        # Valid transaction
        transaction = {
            "amount": 5000,
            "hour": 14,
            "day_of_week": 3,
            "is_weekend": 0,
            "is_night": 0,
            "merchant_id": "MERCH_001",
            "device_id": "DEV_12345",
            "location_lat": 28.6139,
            "location_lon": 77.2090,
            "transaction_velocity": 2,
            "failed_attempts": 0,
            "amount_deviation_pct": 15.5
        }
        
        response = requests.post(f"{BASE_URL}/api/predict_ml", json=transaction)
        data = response.json()
        success = (
            response.status_code == 200 and
            'risk_score' in data and
            'risk_level' in data and
            'action' in data and
            0 <= data['risk_score'] <= 100
        )
        print_test("ML Prediction (Valid)", success,
                  f"Risk: {data.get('risk_level', 'N/A')} ({data.get('risk_score', 'N/A')}) -> {data.get('action', 'N/A')}")
        return success
    except Exception as e:
        print_test("ML Prediction (Valid)", False, str(e))
        return False

def test_ml_prediction_suspicious():
    """Test ML prediction with suspicious transaction"""
    try:
        # Highly suspicious transaction
        transaction = {
            "amount": 45000,
            "hour": 2,
            "day_of_week": 1,
            "is_weekend": 0,
            "is_night": 1,
            "merchant_id": "MERCH_999",
            "device_id": "DEV_99999",
            "location_lat": 12.9716,
            "location_lon": 77.5946,
            "transaction_velocity": 8,
            "failed_attempts": 3,
            "amount_deviation_pct": 150.0
        }
        
        response = requests.post(f"{BASE_URL}/api/predict_ml", json=transaction)
        data = response.json()
        success = response.status_code == 200
        print_test("ML Prediction (Suspicious)", success,
                  f"Risk: {data.get('risk_level', 'N/A')} ({data.get('risk_score', 'N/A')}) -> {data.get('action', 'N/A')}")
        return success
    except Exception as e:
        print_test("ML Prediction (Suspicious)", False, str(e))
        return False

def test_ml_prediction_missing_fields():
    """Test ML prediction with missing required fields"""
    try:
        # Missing fields
        transaction = {
            "amount": 1000,
            "hour": 10
        }
        
        response = requests.post(f"{BASE_URL}/api/predict_ml", json=transaction)
        # Should return 400 or handle gracefully
        success = response.status_code in [200, 400, 422]
        print_test("ML Prediction (Missing Fields)", success,
                  f"Status: {response.status_code}, Response: {response.json().get('error', 'Handled')}")
        return success
    except Exception as e:
        print_test("ML Prediction (Missing Fields)", False, str(e))
        return False

def test_ml_prediction_invalid_types():
    """Test ML prediction with invalid data types"""
    try:
        # Invalid data types
        transaction = {
            "amount": "invalid",
            "hour": "not_a_number",
            "day_of_week": 3,
            "is_weekend": 0,
            "is_night": 0
        }
        
        response = requests.post(f"{BASE_URL}/api/predict_ml", json=transaction)
        success = response.status_code in [400, 422]
        print_test("ML Prediction (Invalid Types)", success,
                  f"Correctly rejected: Status {response.status_code}")
        return success
    except Exception as e:
        print_test("ML Prediction (Invalid Types)", False, str(e))
        return False

def test_recent_predictions():
    """Test recent predictions endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/recent_predictions")
        data = response.json()
        success = (
            response.status_code == 200 and
            isinstance(data, list)
        )
        print_test("Recent Predictions", success,
                  f"Count: {len(data)}")
        return success
    except Exception as e:
        print_test("Recent Predictions", False, str(e))
        return False

def test_model_performance():
    """Test model performance endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/api/model_performance")
        data = response.json()
        success = (
            response.status_code == 200 and
            isinstance(data, list) and
            len(data) > 0
        )
        if success and data:
            best_model = data[0]
            print_test("Model Performance", success,
                      f"Best: {best_model.get('model', 'N/A')}, "
                      f"F1: {best_model.get('f1_score', 'N/A')}")
        else:
            print_test("Model Performance", success)
        return success
    except Exception as e:
        print_test("Model Performance", False, str(e))
        return False

def test_concurrent_requests():
    """Test multiple concurrent requests"""
    try:
        import concurrent.futures
        
        transaction = {
            "amount": 1000,
            "hour": 12,
            "day_of_week": 2,
            "is_weekend": 0,
            "is_night": 0,
            "transaction_velocity": 1,
            "failed_attempts": 0
        }
        
        def make_request():
            return requests.post(f"{BASE_URL}/api/predict_ml", json=transaction)
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(make_request) for _ in range(10)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
        success = all(r.status_code == 200 for r in results)
        print_test("Concurrent Requests (10)", success,
                  f"All {len(results)} requests successful")
        return success
    except Exception as e:
        print_test("Concurrent Requests", False, str(e))
        return False

def test_response_time():
    """Test API response time"""
    try:
        transaction = {
            "amount": 5000,
            "hour": 14,
            "day_of_week": 3,
            "is_weekend": 0,
            "is_night": 0,
            "transaction_velocity": 2,
            "failed_attempts": 0
        }
        
        start = time.time()
        response = requests.post(f"{BASE_URL}/api/predict_ml", json=transaction)
        duration = (time.time() - start) * 1000  # ms
        
        success = response.status_code == 200 and duration < 1000  # < 1 second
        print_test("Response Time", success,
                  f"{duration:.2f}ms (threshold: 1000ms)")
        return success
    except Exception as e:
        print_test("Response Time", False, str(e))
        return False

def test_llm_prediction():
    """Test LLM prediction endpoint"""
    try:
        # Simple transaction for LLM analysis
        transaction = {
            "amount": 95000,
            "hour": 3,
            "day_of_week": 6,
            "merchant_id": "SUSPICIOUS_STORE"
        }
        
        print("  (Waiting for LLM response... this may take a few seconds)")
        response = requests.post(f"{BASE_URL}/api/predict_llm", json=transaction)
        data = response.json()
        
        success = response.status_code == 200
        msg = f"Risk: {data.get('risk_level', 'N/A')}, Explanation: {(data.get('explanation', 'N/A') or '')[:60]}..." if success else f"Error: {data.get('error', 'Unknown')}"
        
        print_test("LLM Prediction", success, msg)
        return success
    except Exception as e:
        print_test("LLM Prediction", False, str(e))
        return False

def run_all_tests():
    """Run all tests"""
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BLUE}UPI FRAUD DETECTION - API TESTS{Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    tests = [
        ("Core Endpoints", [
            test_health_check,
            test_stats_endpoint,
            test_model_performance,
            test_recent_predictions,
        ]),
        ("ML Predictions", [
            test_ml_prediction_valid,
            test_ml_prediction_suspicious,
            test_ml_prediction_missing_fields,
            test_ml_prediction_invalid_types,
            test_llm_prediction,
        ]),
        ("Performance & Load", [
            test_response_time,
            test_concurrent_requests,
        ])
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for category, test_funcs in tests:
        print(f"\n{Colors.BLUE}[{category}]{Colors.END}")
        print("-" * 60)
        
        for test_func in test_funcs:
            total_tests += 1
            if test_func():
                passed_tests += 1
            time.sleep(0.1)  # Small delay between tests
    
    # Summary
    print(f"\n{Colors.BLUE}{'='*60}{Colors.END}")
    success_rate = (passed_tests / total_tests * 100) if total_tests > 0 else 0
    color = Colors.GREEN if success_rate == 100 else (Colors.YELLOW if success_rate >= 80 else Colors.RED)
    print(f"{color}RESULTS: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%){Colors.END}")
    print(f"{Colors.BLUE}{'='*60}{Colors.END}\n")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    try:
        success = run_all_tests()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Tests interrupted by user{Colors.END}")
        exit(1)
    except Exception as e:
        print(f"\n{Colors.RED}Test suite failed: {e}{Colors.END}")
        exit(1)
