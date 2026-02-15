"""
Transaction validation (SRP: single responsibility for input validation).
"""


def validate_transaction_data(data: dict) -> dict:
    """Validate and clean transaction data for prediction endpoints."""
    required_fields = ['amount', 'hour', 'day_of_week']
    missing = [f for f in required_fields if f not in data]
    if missing:
        raise ValueError(f"Missing required fields: {', '.join(missing)}")

    try:
        amount = float(data['amount'])
        if amount < 0:
            raise ValueError("Amount must be positive")
        if amount > 1000000:
            raise ValueError("Amount exceeds maximum limit (1,000,000)")
    except (TypeError, ValueError) as e:
        raise ValueError(f"Invalid amount: {str(e)}")

    try:
        hour = int(data['hour'])
        if not 0 <= hour <= 23:
            raise ValueError("Hour must be between 0 and 23")
    except (TypeError, ValueError):
        raise ValueError("Invalid hour format")

    try:
        day = int(data['day_of_week'])
        if not 0 <= day <= 6:
            raise ValueError("Day of week must be between 0 and 6")
    except (TypeError, ValueError):
        raise ValueError("Invalid day_of_week format")

    defaults = {
        'is_weekend': 1 if data.get('day_of_week', 0) >= 5 else 0,
        'is_night': 1 if data.get('hour', 12) in [22, 23, 0, 1, 2, 3, 4, 5] else 0,
        'transaction_velocity': 1, 'failed_attempts': 0, 'amount_deviation_pct': 0.0,
        'user_id': 'UNKNOWN', 'merchant_id': 'UNKNOWN', 'device_id': 'UNKNOWN',
        'location_lat': 28.6139, 'location_lon': 77.2090,
        'reversed_attempts': 0, 'device_change': 0, 'location_change_km': 0.0,
        'is_new_beneficiary': 0, 'beneficiary_fan_in': 1,
        'beneficiary_account_age_days': 365, 'approval_delay_sec': 1.0,
        'is_multiple_device_accounts': 0, 'network_switch': 0,
        'recent_unknown_call': 0, 'has_suspicious_keywords': 0,
        'merchant_category_mismatch': 0, 'fraud_network_proximity': 0.0
    }
    return {**defaults, **data}
