"""
UPI Fraud Detection API - Production Ready
Flask REST API with comprehensive error handling,logging, and validation
"""

import os
import sys
import logging
from datetime import datetime
from functools import wraps
import traceback

from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np

# Add src to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config import config

# Project root for resolving data/results paths (works from any cwd)
_APP_BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Configure logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=getattr(logging, config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(config.LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

app = Flask(
    __name__,
    static_folder=os.path.join(_APP_BASE, 'web', 'static'),
    template_folder=os.path.join(_APP_BASE, 'web', 'templates')
)

# CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": config.CORS_ORIGINS.split(',') if config.CORS_ORIGINS != '*' else '*'
    }
})

# In-memory storage (replace with database in production)
recent_predictions = []
MAX_RECENT_PREDICTIONS = 100

# Rate limiting storage
from collections import defaultdict, deque
request_counts = defaultdict(lambda: deque())

def rate_limit(max_per_minute=60):
    """Rate limiting decorator"""
    def decorator(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if not config.RATE_LIMIT_ENABLED:
                return f(*args, **kwargs)
            
            # Get client IP
            client_ip = request.remote_addr
            now = datetime.now().timestamp()
            minute_ago = now - 60
            
            # Clean old requests
            request_counts[client_ip] = deque(
                [req_time for req_time in request_counts[client_ip] if req_time > minute_ago]
            )
            
            # Check rate limit
            if len(request_counts[client_ip]) >= max_per_minute:
                logger.warning(f"Rate limit exceeded for {client_ip}")
                return jsonify({
                    'error': 'Rate limit exceeded',
                    'message': f'Maximum {max_per_minute} requests per minute'
                }), 429
            
            # Add current request
            request_counts[client_ip].append(now)
            
            return f(*args, **kwargs)
        return wrapped
    return decorator

def require_api_key(f):
    """API key authentication decorator"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not config.API_KEY_REQUIRED:
            return f(*args, **kwargs)
        
        api_key = request.headers.get('X-API-Key')
        if not api_key or api_key not in config.API_KEYS:
            logger.warning(f"Invalid API key attempted from {request.remote_addr}")
            return jsonify({'error': 'Invalid or missing API key'}), 401
        
        return f(*args, **kwargs)
    return decorated_function

def handle_errors(f):
    """Global error handler decorator"""
    @wraps(f)
    def wrapped(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except ValueError as e:
            logger.error(f"Validation error: {str(e)}")
            return jsonify({
                'error': 'Validation error',
                'message': str(e)
            }), 400
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}\\n{traceback.format_exc()}")
            return jsonify({
                'error': 'Internal server error',
                'message': str(e) if config.DEBUG else 'An error occurred'
            }), 500
    return wrapped

# Load models and preprocessor
try:
    logger.info("Loading models and preprocessor...")
    ml_model = joblib.load(config.MODEL_PATH)
    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    logger.info(f"✓ ML model loaded: {config.MODEL_PATH}")
    logger.info(f"✓ Preprocessor loaded: {config.PREPROCESSOR_PATH}")
except Exception as e:
    logger.error(f"Failed to load models: {str(e)}")
    ml_model = None
    preprocessor = None

# Load LLM if enabled
llm_detector = None
llm_init_error = None

# Log that API key is set (never log the key itself)
if config.GROQ_API_KEY:
    logger.info("API key detected.")

if config.LLM_ENABLED:
    try:
        # Check if groq is installed
        import importlib.util
        if importlib.util.find_spec("groq") is None:
            raise ImportError("The 'groq' library is not installed. Please run: pip install groq")
            
        from llm_detector import LLMFraudDetector
        llm_detector = LLMFraudDetector(api_key=config.GROQ_API_KEY, model=config.LLM_MODEL)
        logger.info(f"✓ LLM detector loaded: {config.LLM_MODEL}")
    except Exception as e:
        llm_init_error = str(e)
        logger.warning(f"LLM detector initialization failed: {str(e)}")

# Load model performance data (CSV has index column = model name, no header for first col)
try:
    import csv
    model_performance = []
    perf_path = os.path.join(_APP_BASE, 'results', 'model_performance.csv')
    _metric_keys = {'accuracy', 'precision', 'recall', 'f1_score', 'roc_auc'}
    with open(perf_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Model name is in the first column (pandas index); DictReader may give key '' or 'Unnamed: 0'
            model_name = row.get('model')
            if not model_name:
                for k in row:
                    if k not in _metric_keys and row.get(k, '').strip():
                        model_name = row[k].strip()
                        break
            if not model_name:
                model_name = 'unknown'
            row['model'] = model_name
            # Ensure numeric fields are numbers for JSON/charts
            for key in _metric_keys:
                if key in row and row[key] not in ('', None):
                    try:
                        row[key] = float(row[key])
                    except (TypeError, ValueError):
                        pass
            model_performance.append(row)
    logger.info(f"✓ Loaded performance data for {len(model_performance)} models")
except Exception as e:
    logger.warning(f"Could not load model performance: {str(e)}")
    model_performance = []

# Load dataset globally for endpoints
try:
    logger.info("Loading dataset...")
    data_path = os.path.join(_APP_BASE, 'data', 'upi_transactions.csv')
    DATASET = pd.read_csv(data_path)
    logger.info(f"✓ Dataset loaded: {len(DATASET)} records")
except Exception as e:
    logger.warning(f"Could not load dataset: {str(e)}")
    DATASET = None

@app.route('/')
def index():
    """Serve the main dashboard"""
    return render_template('index.html')

@app.route('/health')
@handle_errors
def health_check():
    """Health check endpoint"""
    status = {
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'models': {
            'ml_model': ml_model is not None,
            'preprocessor': preprocessor is not None,
            'llm_detector': llm_detector is not None
        },
        'config': {
            'debug': config.DEBUG,
            'rate_limiting': config.RATE_LIMIT_ENABLED,
            'llm_enabled': config.LLM_ENABLED
        }
    }
    return jsonify(status)

@app.route('/api/stats')
@handle_errors
@rate_limit(max_per_minute=config.RATE_LIMIT_PER_MINUTE)
def get_stats():
    """Get fraud detection statistics"""
    if DATASET is None:
        return jsonify({
            'total_transactions': 0,
            'fraud_count': 0,
            'legitimate_count': 0,
            'fraud_rate': 0.0,
            'message': 'Dataset not available'
        })
    
    try:
        df = DATASET
        
        stats = {
            'total_transactions': len(df),
            'fraud_count': int(df['is_fraud'].sum()),
            'legitimate_count': int((df['is_fraud'] == 0).sum()),
            'fraud_rate': round(df['is_fraud'].mean() * 100, 2),
            'fraud_percentage': round(df['is_fraud'].mean() * 100, 2),
            'avg_transaction_amount': round(df['amount'].mean(), 2),
            'max_transaction_amount': round(df['amount'].max(), 2),
            'total_fraud_amount': round(df[df['is_fraud'] == 1]['amount'].sum(), 2) if 'amount' in df.columns else 0
        }
        
        return jsonify(stats)
    except Exception as e:
        logger.error(f"Error loading stats: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/hourly_fraud')
@handle_errors
def get_hourly_fraud():
    """Get fraud distribution by hour"""
    if DATASET is None:
        return jsonify({'hours': [], 'fraud_rate': []})
    
    try:
        # Group by hour and calculate fraud rate
        hourly = DATASET.groupby('hour')['is_fraud'].mean() * 100
        hourly = hourly.reindex(range(24), fill_value=0)
        
        return jsonify({
            'hours': list(hourly.index),
            'fraud_rate': [round(x, 2) for x in hourly.values]
        })
    except Exception as e:
        logger.error(f"Error calculating hourly fraud: {str(e)}")
        return jsonify({'error': str(e)}), 500

def _get_inner_model(model):
    """Get the inner sklearn model (FraudDetectionModel wraps it in .model)."""
    if model is None:
        return None
    return getattr(model, 'model', model)

@app.route('/api/feature_importance')
@handle_errors
def get_feature_importance():
    """Get model feature importance"""
    inner = _get_inner_model(ml_model)
    if inner is None or not hasattr(inner, 'feature_importances_'):
        # Return mock/default importance if not available
        return jsonify({
            'features': ['beneficiary_fan_in', 'transaction_velocity', 'amount', 'is_night', 'failed_attempts'],
            'importance': [0.3, 0.25, 0.2, 0.15, 0.1]
        })
    
    try:
        importance = inner.feature_importances_
        # Get feature names from preprocessor if available
        feature_names = preprocessor.feature_names if preprocessor and hasattr(preprocessor, 'feature_names') else [f'Feature {i}' for i in range(len(importance))]
        
        # Sort by importance
        indices = np.argsort(importance)[::-1][:10]  # Top 10
        
        return jsonify({
            'features': [feature_names[i] for i in indices],
            'importance': [round(float(importance[i]), 3) for i in indices]
        })
    except Exception as e:
        logger.error(f"Error getting feature importance: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/recent_transactions')
@handle_errors
def get_recent_transactions():
    """Get recent transactions from dataset"""
    if DATASET is None:
        return jsonify([])
    
    try:
        # Return sample of last 20 transactions
        # Ensure we have required columns for display
        cols = ['transaction_id', 'amount', 'hour', 'transaction_velocity', 'device_change', 'location_change_km', 'is_fraud']
        
        # Check which columns actually exist
        available_cols = [c for c in cols if c in DATASET.columns]
        
        recent_df = DATASET.tail(20)[available_cols].fillna(0)
        recent = recent_df.to_dict(orient='records')
        return jsonify(recent)
    except Exception as e:
        logger.error(f"Error getting recent transactions: {str(e)}")
        return jsonify([])

@app.route('/api/llm_samples')
@handle_errors
def get_llm_samples():
    """Get LLM training samples with reasoning"""
    try:
        results_path = os.path.join(_APP_BASE, 'results', 'llm_predictions.csv')
        if not os.path.exists(results_path):
            return jsonify([])
        df = pd.read_csv(results_path)
        if 'llm_reasoning' not in df.columns or 'llm_prediction' not in df.columns:
            return jsonify([])
        # Build response with keys the dashboard expects (llm_risk_factors = risk_factors from CSV)
        samples = []
        for _, row in df.head(5).iterrows():
            rec = row.to_dict()
            rec['llm_risk_factors'] = rec.get('risk_factors', '[]')
            samples.append(rec)
        return jsonify(samples)
    except Exception as e:
        logger.error(f"Error loading LLM samples: {str(e)}")
        return jsonify([])

def validate_transaction_data(data: dict) -> dict:
    """Validate and clean transaction data"""
    required_fields = ['amount', 'hour', 'day_of_week']
    
    # Check required fields
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")
    
    # Validate amount
    try:
        amount = float(data['amount'])
        if amount < 0:
            raise ValueError("Amount must be positive")
        if amount > 1000000:
            raise ValueError("Amount exceeds maximum limit (1,000,000)")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid amount: {str(e)}")
    
    # Validate hour
    try:
        hour = int(data['hour'])
        if not 0 <= hour <= 23:
            raise ValueError("Hour must be between 0 and 23")
    except (TypeError, ValueError):
        raise ValueError("Invalid hour format")
    
    # Validate day_of_week
    try:
        day = int(data['day_of_week'])
        if not 0 <= day <= 6:
            raise ValueError("Day of week must be between 0 and 6")
    except (TypeError, ValueError):
        raise ValueError("Invalid day_of_week format")
    
    # Set defaults for optional fields
    defaults = {
        'is_weekend': 1 if data.get('day_of_week', 0) >= 5 else 0,
        'is_night': 1 if data.get('hour', 12) in [22, 23, 0, 1, 2, 3, 4, 5] else 0,
        'transaction_velocity': 1,
        'failed_attempts': 0,
        'amount_deviation_pct': 0.0,
        'user_id': 'UNKNOWN',
        'merchant_id': 'UNKNOWN',
        'device_id': 'UNKNOWN',
        'location_lat': 28.6139,
        'location_lon': 77.2090,
        
        # Enhanced features defaults
        'reversed_attempts': 0,
        'device_change': 0,
        'location_change_km': 0.0,
        'is_new_beneficiary': 0,
        'beneficiary_fan_in': 1,
        'beneficiary_account_age_days': 365,
        'approval_delay_sec': 1.0,
        'is_multiple_device_accounts': 0,
        'network_switch': 0,
        'recent_unknown_call': 0,
        'has_suspicious_keywords': 0,
        'merchant_category_mismatch': 0,
        'fraud_network_proximity': 0.0
    }
    
    # Merge with defaults
    clean_data = {**defaults, **data}
    
    return clean_data

@app.route('/api/predict', methods=['POST'])
@app.route('/api/predict_ml', methods=['POST'])
@handle_errors
@require_api_key
@rate_limit(max_per_minute=config.RATE_LIMIT_PER_MINUTE)
def predict_ml():
    """ML-based fraud prediction"""
    if ml_model is None or preprocessor is None:
        return jsonify({'error': 'ML model not available'}), 503
    
    data = request.get_json()
    if not data:
        raise ValueError("No data provided")
    
    # Validate and clean data
    clean_data = validate_transaction_data(data)
    
    # Create DataFrame
    df = pd.DataFrame([clean_data])
    
    # Preprocess (Engineer features and encode inputs)
    processed_df = preprocessor.preprocess(df, fit=False)
    
    # Prepare features (Scale and align columns)
    X, _ = preprocessor.prepare_features(processed_df, fit=False)
    
    # Predict
    prediction = int(ml_model.predict(X)[0])
    confidence = float(ml_model.predict_proba(X)[0] * 100)
    
    result = {
        'prediction': 'fraud' if prediction == 1 else 'legitimate',
        'confidence': round(confidence, 2),
        'probability': round(confidence / 100, 4),
        'model': 'Random Forest',
        'timestamp': datetime.now().isoformat(),
        'transaction_data': {
            'amount': clean_data['amount'],
            'hour': clean_data['hour'],
            'day_of_week': clean_data['day_of_week']
        }
    }
    
    #Store prediction
    recent_predictions.append(result)
    if len(recent_predictions) > MAX_RECENT_PREDICTIONS:
        recent_predictions.pop(0)
    
    logger.info(f"ML Prediction: {result['prediction']} ({result['confidence']}%)")
    
    return jsonify(result)

@app.route('/api/predict_llm', methods=['POST'])
@handle_errors
@require_api_key
@rate_limit(max_per_minute=10)  # Lower rate limit for LLM
def predict_llm():
    """LLM-based fraud prediction"""
    if llm_detector is None:
        # Diagnostic info
        error_msg = "LLM detector not available."
        if not config.LLM_ENABLED:
            error_msg += " (LLM_ENABLED is False in config)"
        else:
            error_msg += f" (Initialization failed: {llm_init_error or 'Check server logs'})"
        return jsonify({'error': error_msg}), 503
    
    data = request.get_json()
    if not data:
        raise ValueError("No data provided")
    
    # Validate data
    clean_data = validate_transaction_data(data)
    
    try:
        prediction, confidence, reasoning, risk_factors = llm_detector.predict_single(clean_data)
        
        result = {
            'prediction': 'fraud' if prediction == 1 else 'legitimate',
            'confidence': round(confidence, 2),
            'reasoning': reasoning,
            'risk_factors': risk_factors,
            'model': 'LLM (Groq)',
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"LLM Prediction: {result['prediction']} ({result['confidence']}%)")
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"LLM prediction failed: {str(e)}")
        return jsonify({
            'error': 'LLM prediction failed',
            'message': str(e)
        }), 500

@app.route('/api/recent_predictions')
@handle_errors
@rate_limit(max_per_minute=config.RATE_LIMIT_PER_MINUTE)
def get_recent_predictions():
    """Get recent predictions"""
    return jsonify(recent_predictions[-20:])  # Last 20

@app.route('/api/model_performance')
@handle_errors
@rate_limit(max_per_minute=config.RATE_LIMIT_PER_MINUTE)
def get_model_performance():
    """Get model performance metrics"""
    return jsonify(model_performance)

@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(e):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {str(e)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    try:
        # Validate configuration (model/preprocessor); start anyway so dashboard is usable
        try:
            config.validate()
            logger.info("Configuration validated successfully")
        except ValueError as e:
            logger.warning(str(e))
            logger.info("Run training first: cd src && python train.py   (or python train.py --with-llm for LLM)")
        
        logger.info(f"Starting UPI Fraud Detection API on {config.HOST}:{config.PORT}")
        logger.info(f"Debug mode: {config.DEBUG}")
        logger.info(f"Rate limiting: {config.RATE_LIMIT_ENABLED}")
        logger.info(f"LLM enabled: {config.LLM_ENABLED}")
        
        app.run(
            host=config.HOST,
            port=config.PORT,
            debug=config.DEBUG
        )
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        sys.exit(1)
