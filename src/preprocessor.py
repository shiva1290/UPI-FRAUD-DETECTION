"""
Data Preprocessing Pipeline for UPI Fraud Detection
Handles data cleaning, feature engineering, and transformation
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib

class UPIPreprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        
    def preprocess(self, data, fit=True):
        """
        Complete preprocessing pipeline
        
        Args:
            data: Input DataFrame
            fit: Whether to fit the preprocessor (True for training data)
        
        Returns:
            Preprocessed DataFrame
        """
        df = data.copy()
        
        # Handle missing values
        df = self._handle_missing_values(df)
        
        # Feature engineering
        df = self._engineer_features(df)
        
        # Encode categorical variables
        df = self._encode_categorical(df, fit=fit)
        
        # Remove duplicates
        df = df.drop_duplicates()
        
        # Detect and handle outliers
        if fit:
            df = self._handle_outliers(df)
        
        return df
    
    def _handle_missing_values(self, df):
        """Handle missing values in the dataset"""
        
        # Fill numerical columns with median
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Fill categorical columns with mode
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def _engineer_features(self, df):
        """Create additional features from existing data"""
        
        # Convert timestamp to datetime if not already
        if 'timestamp' in df.columns and df['timestamp'].dtype == 'object':
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Temporal features - only if not already present
        if 'timestamp' in df.columns and 'hour' not in df.columns:
            df['hour'] = df['timestamp'].dt.hour
            df['day_of_week'] = df['timestamp'].dt.dayofweek
            df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
            df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        # Amount-based features - only if not already present
        if 'amount' in df.columns and 'amount_log' not in df.columns:
            df['amount_log'] = np.log1p(df['amount'])
            df['is_high_amount'] = (df['amount'] > df['amount'].quantile(0.95)).astype(int)
            df['is_round_amount'] = (df['amount'] % 100 == 0).astype(int)
        
        # Velocity features - only if not already present
        if 'transaction_velocity' in df.columns and 'is_high_velocity' not in df.columns:
            df['is_high_velocity'] = (df['transaction_velocity'] > 5).astype(int)
        
        # Location features - only if not already present
        if 'location_change_km' in df.columns and 'is_location_jump' not in df.columns:
            df['is_location_jump'] = (df['location_change_km'] > 50).astype(int)
        
        # Interaction features - only if not already present
        if 'failed_attempts' in df.columns and 'transaction_velocity' in df.columns and 'risk_score' not in df.columns:
            df['risk_score'] = df['failed_attempts'] * df['transaction_velocity']
        
        return df
    
    def _encode_categorical(self, df, fit=True):
        """Encode categorical variables"""
        
        categorical_cols = ['user_id', 'merchant_id', 'device_id']
        
        for col in categorical_cols:
            if col in df.columns:
                if fit:
                    le = LabelEncoder()
                    df[col + '_encoded'] = le.fit_transform(df[col].astype(str))
                    self.label_encoders[col] = le
                else:
                    if col in self.label_encoders:
                        # Handle unseen categories
                        le = self.label_encoders[col]
                        df[col + '_encoded'] = df[col].apply(
                            lambda x: le.transform([str(x)])[0] if str(x) in le.classes_ else -1
                        )
        
        return df
    
    def _handle_outliers(self, df, threshold=3):
        """Detect and handle outliers using IQR method"""
        
        numerical_cols = ['amount', 'transaction_velocity', 'location_change_km']
        
        for col in numerical_cols:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - threshold * IQR
                upper_bound = Q3 + threshold * IQR
                
                # Cap outliers instead of removing (for fraud detection)
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        return df
    
    def prepare_features(self, df, target_col='is_fraud', scale=True, fit=True):
        """
        Prepare features for model training
        
        Args:
            df: Preprocessed DataFrame
            target_col: Name of target column
            scale: Whether to scale features
            fit: Whether to fit the scaler (True for training, False for prediction)
        
        Returns:
            X, y: Feature matrix and target vector
        """
        # Remove non-feature columns
        exclude_cols = [
            target_col, 'transaction_id', 'timestamp', 
            'user_id', 'merchant_id', 'device_id'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        # Handle any remaining missing values before scaling
        # Fill numerical columns with median
        for col in X.columns:
            if X[col].dtype in [np.float64, np.int64]:
                if X[col].isnull().any():
                    fill_value = X[col].median()
                    X.loc[:, col] = X[col].fillna(fill_value)
        
        # Verify no NaN values remain
        if X.isnull().any().any():
            print(f"⚠️  Warning: NaN values still present, dropping them")
            X = X.fillna(0)  # Fallback: fill with 0
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
        else:
            # Ensure columns match the expected features
            if self.feature_names is not None:
                # Reorder columns to match training order
                X = X[self.feature_names]
        
        # Scale features
        if scale:
            if fit:
                X_scaled = self.scaler.fit_transform(X.values)  # Use .values to get numpy array
            else:
                X_scaled = self.scaler.transform(X.values)
            X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X, y
    
    def save(self, path='../models/preprocessor.pkl'):
        """Save preprocessor to disk"""
        joblib.dump(self, path)
        print(f"✓ Preprocessor saved to {path}")
    
    @staticmethod
    def load(path='../models/preprocessor.pkl'):
        """Load preprocessor from disk"""
        return joblib.load(path)


if __name__ == "__main__":
    # Test preprocessing pipeline
    print("Testing UPI Preprocessor...")
    
    # Load sample data
    data = pd.read_csv('../data/upi_transactions.csv')
    
    # Initialize and run preprocessor
    preprocessor = UPIPreprocessor()
    processed_data = preprocessor.preprocess(data, fit=True)
    
    print(f"\nOriginal shape: {data.shape}")
    print(f"Processed shape: {processed_data.shape}")
    
    # Prepare features
    X, y = preprocessor.prepare_features(processed_data)
    
    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target vector shape: {y.shape}")
    print(f"\nFeatures: {preprocessor.feature_names}")
    
    # Save preprocessor
    preprocessor.save()
