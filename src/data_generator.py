"""
Enhanced UPI Data Generator with Real Dataset Integration
Combines real Kaggle data with improved synthetic generation
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
import os

class EnhancedUPIDataGenerator:
    def __init__(self, use_real_data=True, target_samples=150000):
        self.use_real_data = use_real_data
        self.target_samples = target_samples
        self.real_data = None
        self.real_fraud_rate = 0.035  # Target 3.5% fraud rate
        
        # Load real dataset if available
        if use_real_data:
            self._load_real_dataset()
        
    def _load_real_dataset(self):
        """Load and prepare the real Kaggle dataset (optional; uses synthetic if missing)."""
        try:
            paths_file = os.path.join(os.path.dirname(__file__), '..', 'data', 'kaggle_dataset_paths.txt')
            if not os.path.exists(paths_file):
                print("⚠️  Kaggle paths file not found, using synthetic data only")
                return
            with open(paths_file, 'r') as f:
                paths = dict(line.strip().split(': ', 1) for line in f if ':' in line)
            
            dataset_path = paths.get('skullagos5246', '')
            csv_path = os.path.join(dataset_path, 'upi_transactions_2024.csv')
            
            if os.path.exists(csv_path):
                self.real_data = pd.read_csv(csv_path)
                print(f"✓ Loaded real dataset: {len(self.real_data)} transactions")
                print(f"  Fraud rate: {self.real_data['fraud_flag'].mean()*100:.2f}%")
            else:
                print("⚠️  Real dataset not found, using synthetic data only")
        except Exception as e:
            print(f"⚠️  Error loading real dataset: {e}")
    
    def generate(self):
        """Generate enhanced dataset with real and synthetic data"""
        
        if self.real_data is not None:
            # Use 50% real data, 50% enhanced synthetic
            num_real = min(len(self.real_data), self.target_samples // 2)
            num_synthetic = self.target_samples - num_real
            
            print(f"\n📊 Generating mixed dataset:")
            print(f"  Real transactions: {num_real}")
            print(f"  Synthetic transactions: {num_synthetic}")
            
            # Sample from real data
            real_sample = self.real_data.sample(n=num_real, random_state=42)
            real_df = self._convert_real_to_schema(real_sample)
            
            # Generate synthetic data with patterns from real data
            synthetic_df = self._generate_realistic_synthetic(num_synthetic)
            
            # Combine
            combined_df = pd.concat([real_df, synthetic_df], ignore_index=True)
            
            # Add noise and ambiguity
            combined_df = self._add_realistic_noise(combined_df)
            
            # Shuffle
            combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)
            
        else:
            # Generate only synthetic with enhanced realism
            combined_df = self._generate_realistic_synthetic(self.target_samples)
            combined_df = self._add_realistic_noise(combined_df)
        
        return combined_df
    
    def _convert_real_to_schema(self, real_df):
        """Convert real dataset to our schema"""
        
        converted = pd.DataFrame()
        
        # Generate transaction IDs
        converted['transaction_id'] = [f"REAL_{i:08d}" for i in range(len(real_df))]
        
        # Convert timestamp
        converted['timestamp'] = pd.to_datetime(real_df['timestamp'].values)
        
        # Amount
        converted['amount'] = real_df['amount (INR)'].values
        
        # Generate user IDs based on age group
        converted['user_id'] = [f"USER_{random.randint(1, 5000)}" for _ in range(len(real_df))]
        
        # Merchant ID from category
        converted['merchant_id'] = [f"MERCH_{random.randint(100, 999)}" for _ in range(len(real_df))]
        
        # Device ID from device type
        converted['device_id'] = [f"DEV_{random.randint(10000, 99999)}" for _ in range(len(real_df))]
        
        # Location (use Indian state coordinates - simplified)
        converted['location_lat'] = np.random.uniform(8, 35, len(real_df))  # India latitude range
        converted['location_lon'] = np.random.uniform(68, 97, len(real_df))  # India longitude range
        
        # Time features
        converted['hour'] = real_df['hour_of_day'].values
        converted['day_of_week'] = real_df['day_of_week'].map({
            'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
            'Friday': 4, 'Saturday': 5, 'Sunday': 6
        }).fillna(0).astype(int)
        converted['is_weekend'] = real_df['is_weekend'].values
        converted['is_night'] = (converted['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5])).astype(int)
        
        # Generate behavioral features based on fraud flag
        is_fraud = real_df['fraud_flag'].values
        
        # Fraud transactions have different patterns
        converted['amount_deviation_pct'] = np.where(
            is_fraud, 
            np.random.uniform(50, 300, len(real_df)),  # High deviation for fraud
            np.random.uniform(0, 50, len(real_df))      # Low deviation for legitimate
        )
        
        converted['transaction_velocity'] = np.where(
            is_fraud,
            np.random.randint(5, 15, len(real_df)),
            np.random.randint(1, 5, len(real_df))
        )
        
        converted['failed_attempts'] = np.where(
            is_fraud,
            np.random.randint(1, 8, len(real_df)),
            np.random.randint(0, 2, len(real_df))
        )
        
        converted['reversed_attempts'] = np.where(
            is_fraud,
            np.random.randint(0, 5, len(real_df)),
            np.random.randint(0, 1, len(real_df))
        )
        
        converted['device_change'] = np.where(
            is_fraud,
            np.random.randint(0, 2, len(real_df)),
            np.random.randint(0, 1, len(real_df))
        )
        
        converted['location_change_km'] = np.where(
            is_fraud,
            np.random.uniform(20, 500, len(real_df)),
            np.random.uniform(0, 30, len(real_df))
        )
        
        converted['is_new_beneficiary'] = np.where(
            is_fraud,
            np.random.randint(0, 2, len(real_df)),
            np.random.randint(0, 1, len(real_df))
        )
        
        converted['beneficiary_fan_in'] = np.where(
            is_fraud,
            np.random.randint(10, 50, len(real_df)),
            np.random.randint(1, 10, len(real_df))
        )
        
        converted['beneficiary_account_age_days'] = np.where(
            is_fraud,
            np.random.randint(1, 60, len(real_df)),
            np.random.randint(30, 1000, len(real_df))
        )
        
        converted['approval_delay_sec'] = np.where(
            is_fraud,
            np.random.uniform(1, 5, len(real_df)),
            np.random.uniform(5, 30, len(real_df))
        )
        
        converted['is_multiple_device_accounts'] = np.where(
            is_fraud,
            np.random.randint(0, 2, len(real_df)),
            0
        )
        
        converted['network_switch'] = np.where(
            is_fraud,
            np.random.randint(0, 2, len(real_df)),
            0
        )
        
        converted['recent_unknown_call'] = np.where(
            is_fraud,
            np.random.randint(0, 2, len(real_df)),
            0
        )
        
        converted['has_suspicious_keywords'] = np.where(
            is_fraud,
            np.random.randint(0, 2, len(real_df)),
            0
        )
        
        converted['merchant_category_mismatch'] = np.where(
            is_fraud,
            np.random.randint(0, 2, len(real_df)),
            0
        )
        
        converted['fraud_network_proximity'] = np.where(
            is_fraud,
            np.random.randint(0, 2, len(real_df)),
            0
        )
        
        # Fraud label
        converted['is_fraud'] = is_fraud
        
        return converted
    
    def _generate_realistic_synthetic(self, n):
        """Generate synthetic data with realistic patterns"""
        
        # Calculate fraud/legitimate split
        num_fraud = int(n * self.real_fraud_rate)
        num_legit = n - num_fraud
        
        print(f"  Generating {num_legit} legitimate + {num_fraud} fraud transactions")
        
        # Generate legitimate transactions
        legit_df = self._generate_legitimate_batch(num_legit)
        
        # Generate fraud transactions with variety
        fraud_df = self._generate_fraud_batch(num_fraud)
        
        # Combine
        df = pd.concat([legit_df, fraud_df], ignore_index=True)
        
        return df
    
    def _generate_legitimate_batch(self, n):
        """Generate realistic legitimate transactions with HIGH overlap with fraud"""
        
        df = pd.DataFrame()
        
        df['transaction_id'] = [f"SYNTH_{i:08d}" for i in range(n)]
        
        # Realistic timestamps (last 6 months)
        start_date = datetime.now() - timedelta(days=180)
        df['timestamp'] = [start_date + timedelta(
            days=random.randint(0, 180),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59)
        ) for _ in range(n)]
        
        # Realistic amounts with MORE large transactions
        amounts = []
        for _ in range(n):
            if random.random() < 0.5:  # 50% small transactions
                amounts.append(random.uniform(10, 1000))
            elif random.random() < 0.8:  # 30% medium
                amounts.append(random.uniform(1000, 5000))
            else:  # 20% large (more overlap with fraud)
                amounts.append(random.uniform(5000, 30000))
        df['amount'] = amounts
        
        df['user_id'] = [f"USER_{random.randint(1, 5000)}" for _ in range(n)]
        df['merchant_id'] = [f"MERCH_{random.randint(100, 999)}" for _ in range(n)]
        df['device_id'] = [f"DEV_{random.randint(10000, 99999)}" for _ in range(n)]
        
        df['location_lat'] = np.random.uniform(8, 35, n)
        df['location_lon'] = np.random.uniform(68, 97, n)
        
        df['hour'] = [ts.hour for ts in df['timestamp']]
        df['day_of_week'] = [ts.dayofweek for ts in df['timestamp']]
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        # More variable behavioral features (LESS clean separation)
        df['amount_deviation_pct'] = np.random.uniform(0, 80, n)  # Wider range
        df['transaction_velocity'] = np.random.randint(1, 8, n)  # Higher velocities
        df['failed_attempts'] = np.random.choice([0, 0, 0, 1, 1, 2], n)  # More failures
        df['reversed_attempts'] = np.random.choice([0, 0, 0, 1], n)  # Some reversals
        df['device_change'] = np.random.choice([0, 0, 0, 1], n)  # Some changes
        df['location_change_km'] = np.random.uniform(0, 60, n)  # Larger range
        df['is_new_beneficiary'] = np.random.choice([0, 0, 1, 1], n)  # More new beneficiaries
        df['beneficiary_fan_in'] = np.random.randint(1, 20, n)  # Higher range
        df['beneficiary_account_age_days'] = np.random.randint(10, 1200, n)  # Include newer accounts
        df['approval_delay_sec'] = np.random.uniform(3, 25, n)  # Lower min delay
        df['is_multiple_device_accounts'] = np.random.choice([0, 0, 0, 1], n)
        df['network_switch'] = np.random.choice([0, 0, 0, 1], n)
        df['recent_unknown_call'] = np.random.choice([0, 0, 0, 1], n)
        df['has_suspicious_keywords'] = 0
        df['merchant_category_mismatch'] = 0
        df['fraud_network_proximity'] = 0
        
        df['is_fraud'] = 0
        
        # CRITICAL: Add MANY legitimate transactions with fraud-like features (30% borderline!)
        num_borderline = int(n * 0.30)  # 30% borderline cases
        borderline_idx = np.random.choice(n, num_borderline, replace=False)
        
        for idx in borderline_idx:
            # Add 3-5 suspicious features but still legitimate
            num_features = random.randint(3, 5)
            suspicious_features = random.sample([
                'is_night', 'amount_deviation_pct', 'failed_attempts',
                'transaction_velocity', 'is_new_beneficiary', 'device_change',
                'location_change_km', 'reversed_attempts'
            ], num_features)
            
            for feature in suspicious_features:
                if feature == 'is_night':
                    df.loc[idx, 'is_night'] = 1
                    df.loc[idx, 'hour'] = random.choice([22, 23, 0, 1, 2, 3])
                elif feature == 'amount_deviation_pct':
                    df.loc[idx, 'amount_deviation_pct'] = random.uniform(100, 200)
                elif feature == 'failed_attempts':
                    df.loc[idx, 'failed_attempts'] = random.randint(2, 5)
                elif feature == 'transaction_velocity':
                    df.loc[idx, 'transaction_velocity'] = random.randint(6, 10)
                elif feature == 'is_new_beneficiary':
                    df.loc[idx, 'is_new_beneficiary'] = 1
                elif feature == 'device_change':
                    df.loc[idx, 'device_change'] = 1
                elif feature == 'location_change_km':
                    df.loc[idx, 'location_change_km'] = random.uniform(60, 150)
                elif feature == 'reversed_attempts':
                    df.loc[idx, 'reversed_attempts'] = random.randint(1, 3)
        
        return df
    
    def _generate_fraud_batch(self, n):
        """Generate realistic fraud transactions with HIGH overlap with legitimate"""
        
        df = pd.DataFrame()
        
        df['transaction_id'] = [f"FRAUD_{i:08d}" for i in range(n)]
        
        # More varied timing - not all at night
        start_date = datetime.now() - timedelta(days=180)
        timestamps = []
        for _ in range(n):
            # 40% at night, 60% during day (more overlap!)
            if random.random() < 0.4:
                hour = random.choice([22, 23, 0, 1, 2, 3, 4, 5])
            else:
                hour = random.randint(6, 21)
            
            timestamps.append(start_date + timedelta(
                days=random.randint(0, 180),
                hours=hour,
                minutes=random.randint(0, 59)
            ))
        df['timestamp'] = timestamps
        
        # More varied amounts - not all huge
        amounts = []
        for _ in range(n):
            if random.random() < 0.25:  # 25% very high
                amounts.append(random.uniform(10000, 50000))
            elif random.random() < 0.50:  # 25% high
                amounts.append(random.uniform(5000, 10000))
            else:  # 50% moderate/small (HARD TO DETECT)
                amounts.append(random.uniform(500, 5000))
        df['amount'] = amounts
        
        df['user_id'] = [f"USER_{random.randint(1, 5000)}" for _ in range(n)]
        df['merchant_id'] = [f"MERCH_{random.randint(100, 999)}" for _ in range(n)]
        df['device_id'] = [f"DEV_{random.randint(10000, 99999)}" for _ in range(n)]
        
        df['location_lat'] = np.random.uniform(8, 35, n)
        df['location_lon'] = np.random.uniform(68, 97, n)
        
        df['hour'] = [ts.hour for ts in df['timestamp']]
        df['day_of_week'] = [ts.dayofweek for ts in df['timestamp']]
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['is_night'] = df['hour'].isin([22, 23, 0, 1, 2, 3, 4, 5]).astype(int)
        
        # MORE MODERATE behavioral features (not all maxed)
        df['amount_deviation_pct'] = np.random.uniform(40, 200, n)  # Lower range
        df['transaction_velocity'] = np.random.randint(3, 12, n)  # Lower min
        df['failed_attempts'] = np.random.randint(0, 8, n)  # Include 0
        df['reversed_attempts'] = np.random.randint(0, 4, n)  # Include 0
        df['device_change'] = np.random.randint(0, 2, n)
        df['location_change_km'] = np.random.uniform(10, 400, n)  # Lower max
        df['is_new_beneficiary'] = np.random.randint(0, 2, n)
        df['beneficiary_fan_in'] = np.random.randint(5, 50, n)  # Overlap with legit
        df['beneficiary_account_age_days'] = np.random.randint(1, 180, n)  # Some older
        df['approval_delay_sec'] = np.random.uniform(1, 15, n)  # Higher max
        df['is_multiple_device_accounts'] = np.random.randint(0, 2, n)
        df['network_switch'] = np.random.randint(0, 2, n)
        df['recent_unknown_call'] = np.random.randint(0, 2, n)
        df['has_suspicious_keywords'] = np.random.randint(0, 2, n)
        df['merchant_category_mismatch'] = np.random.randint(0, 2, n)
        df['fraud_network_proximity'] = np.random.randint(0, 2, n)
        
        df['is_fraud'] = 1
        
        # CRITICAL: Make 40% of frauds VERY subtle (looks almost legitimate)
        num_subtle = int(n * 0.40)  # 40% subtle fraud
        subtle_idx = np.random.choice(n, num_subtle, replace=False)
        
        for idx in subtle_idx:
            # Make it look VERY legitimate - only 2-3 fraud indicators
            df.loc[idx, 'failed_attempts'] = random.choice([0, 0, 1, 1])
            df.loc[idx, 'transaction_velocity'] = random.randint(2, 5)
            df.loc[idx, 'amount_deviation_pct'] = random.uniform(20, 70)
            df.loc[idx, 'is_night'] = random.choice([0, 0, 1])
            df.loc[idx, 'location_change_km'] = random.uniform(5, 40)
            df.loc[idx, 'reversed_attempts'] = 0
            df.loc[idx, 'approval_delay_sec'] = random.uniform(8, 20)
        
        return df
    
    def _add_realistic_noise(self, df):
        """Add realistic noise, missing values, and LABEL NOISE"""
        
        print("\n  Adding realistic noise patterns...")
        
        # Add missing values (8-12%) - MORE noise
        noise_rate = 0.10
        
        # Columns that can have missing values
        nullable_cols = ['approval_delay_sec', 'location_change_km', 'beneficiary_account_age_days']
        
        for col in nullable_cols:
            if col in df.columns:
                mask = np.random.random(len(df)) < noise_rate
                df.loc[mask, col] = np.nan
        
        # Add small random noise to continuous features
        continuous_cols = ['amount', 'location_lat', 'location_lon', 'amount_deviation_pct']
        
        for col in continuous_cols:
            if col in df.columns:
                noise = np.random.normal(0, df[col].std() * 0.03, len(df))  # More noise
                df[col] = df[col] + noise
                df[col] = df[col].clip(lower=0)  # No negative values
        
        # CRITICAL: Add LABEL NOISE (2% mis-labeled)
        label_noise_rate = 0.02
        num_flips = int(len(df) * label_noise_rate)
        flip_idx = np.random.choice(len(df), num_flips, replace=False)
        df.loc[flip_idx, 'is_fraud'] = 1 - df.loc[flip_idx, 'is_fraud']
        
        print(f"  ✓ Added {noise_rate*100:.1f}% missing values")
        print(f"  ✓ Added Gaussian noise to continuous features")
        print(f"  ✓ Added {label_noise_rate*100:.1f}% label noise (mis-labeled transactions)")
        
        return df


def main():
    """Generate enhanced dataset"""
    
    print("="*70)
    print("ENHANCED UPI DATA GENERATION")
    print("="*70)
    
    generator = EnhancedUPIDataGenerator(
        use_real_data=True,
        target_samples=150000
    )
    
    df = generator.generate()
    
    print(f"\n{'='*70}")
    print("DATASET SUMMARY")
    print(f"{'='*70}")
    print(f"Total transactions: {len(df)}")
    print(f"Fraud transactions: {df['is_fraud'].sum()}")
    print(f"Legitimate transactions: {(~df['is_fraud'].astype(bool)).sum()}")
    print(f"Fraud rate: {df['is_fraud'].mean() * 100:.2f}%")
    
    # Save
    output_path = '../data/upi_transactions.csv'
    df.to_csv(output_path, index=False)
    print(f"\n✓ Dataset saved to: {output_path}")
    
    print(f"\n{'='*70}")
    print("GENERATION COMPLETE")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
