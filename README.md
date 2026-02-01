# UPI Fraud Detection System

A comprehensive machine learning-based fraud detection system for Unified Payments Interface (UPI) transactions.

## Project Structure

```
upi-fraud-detection/
├── data/                   # Dataset storage
├── models/                 # Saved models and preprocessors
├── results/                # Performance metrics and visualizations
├── src/                    # Source code
│   ├── data_generator.py  # Synthetic data generation
│   ├── preprocessor.py    # Data preprocessing pipeline
│   ├── models.py          # ML model implementations
│   └── train.py           # Main training pipeline
├── notebooks/              # Jupyter notebooks for analysis
├── config/                 # Configuration files
└── requirements.txt        # Python dependencies
```

## Features

### Data Generation
- Synthetic UPI transaction dataset with realistic fraud patterns
- Configurable fraud ratio and sample size
- Behavioral features: transaction velocity, device changes, location jumps
- Temporal patterns: time-of-day, day-of-week distributions

### Preprocessing
- Missing value imputation
- Outlier detection and handling (IQR method)
- Feature engineering:
  - Temporal features (hour, day, weekend, night)
  - Amount-based features (log transform, high amount flags)
  - Behavioral features (velocity, location changes)
  - Interaction features (risk scores)
- Categorical encoding
- Feature scaling (StandardScaler)

### Machine Learning Models
- **Logistic Regression**: Baseline linear model
- **Random Forest**: Ensemble tree-based classifier
- **XGBoost**: Gradient boosting with class imbalance handling
- **SVM**: Support Vector Machine with RBF kernel
- **Gradient Boosting**: Adaptive boosting classifier

### LLM-Based Detection (NEW!)
- **Groq API Integration**: Uses Llama 3.3 70B for intelligent fraud analysis
- **Explainable AI**: Provides human-readable reasoning for each prediction
- **Risk Factor Identification**: Identifies specific suspicious patterns
- **Hybrid Approach**: Combines ML speed with LLM explainability

See [LLM_SETUP.md](LLM_SETUP.md) for detailed setup instructions.

### Class Imbalance Handling
- Class weight adjustment
- SMOTE (Synthetic Minority Over-sampling Technique)
- Configurable sampling strategies

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix
- Classification Report

### Visualizations
- Model performance comparison charts
- Confusion matrices
- ROC curves
- Feature importance plots

## Installation

1. **Clone or navigate to the project directory:**
```bash
cd /Users/shivagupta/.gemini/antigravity/scratch/upi-fraud-detection
```

2. **Create a virtual environment (recommended):**
```bash
python3 -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

## Usage

### Quick Start - Run Complete Pipeline

```bash
cd src
python train.py
```

This will:
1. Generate 50,000 synthetic UPI transactions (2% fraud rate)
2. Preprocess and engineer features
3. Train 5 different ML models with SMOTE
4. Evaluate all models and compare performance
5. Save the best model and generate visualizations
6. Export results to CSV

### Individual Components

#### 1. Generate Data Only
```python
from data_generator import UPIDataGenerator

generator = UPIDataGenerator(n_samples=50000, fraud_ratio=0.02)
data = generator.save_data('../data/upi_transactions.csv')
```

#### 2. Preprocess Data
```python
from preprocessor import UPIPreprocessor
import pandas as pd

data = pd.read_csv('../data/upi_transactions.csv')
preprocessor = UPIPreprocessor()
processed_data = preprocessor.preprocess(data, fit=True)
X, y = preprocessor.prepare_features(processed_data)
```

#### 3. Train Specific Model
```python
from models import FraudDetectionModel

model = FraudDetectionModel(model_type='random_forest')
model.train(X_train, y_train, use_smote=True)
metrics = model.evaluate(X_test, y_test)
model.save('../models/my_model.pkl')
```

#### 4. Compare All Models
```python
from models import ModelComparison

comparison = ModelComparison()
comparison.train_all_models(X_train, y_train, use_smote=True)
comparison.evaluate_all_models(X_test, y_test)
best_model, best_name = comparison.get_best_model(metric='f1_score')
```

## Output Files

After running the pipeline, you'll find:

### Models (`models/`)
- `preprocessor.pkl` - Fitted data preprocessor
- `best_model_<name>.pkl` - Best performing model

### Results (`results/`)
- `model_performance.csv` - Performance metrics for all models
- `model_comparison.png` - Visual comparison of all models
- `confusion_matrix_<name>.png` - Confusion matrix for best model
- `roc_curve_<name>.png` - ROC curve for best model

### Data (`data/`)
- `upi_transactions.csv` - Generated transaction dataset

## Using Results in Your Paper

The system generates ready-to-use results for your research paper:

1. **Tables**: Use `model_performance.csv` to create performance comparison tables
2. **Figures**: Include the generated PNG visualizations
3. **Metrics**: All standard fraud detection metrics are computed
4. **Feature Importance**: Top features influencing fraud detection

## Configuration

Modify parameters in the code:

```python
# Data generation
UPIDataGenerator(
    n_samples=50000,      # Number of transactions
    fraud_ratio=0.02,     # 2% fraud rate
    random_state=42       # For reproducibility
)

# Model training
FraudDetectionModel(
    model_type='random_forest',  # Model type
    class_weight='balanced',     # Class weighting
    random_state=42
)
```

## Performance Expectations

With the default configuration (50,000 samples, 2% fraud):

- **Training Time**: 2-5 minutes (all models)
- **Expected F1-Score**: 0.85-0.95 (varies by model)
- **Expected ROC-AUC**: 0.90-0.98

## Research Paper Integration

This implementation directly supports your paper sections:

### Chapter 3: Design Flow/Process
- Explains data generation, preprocessing, and feature engineering
- Documents model selection and training procedures

### Chapter 4: Results Analysis and Validation
- Provides concrete performance metrics
- Includes comparison of multiple approaches
- Demonstrates effectiveness through visualizations

## Citations

When using this implementation in your research, cite the following papers referenced in your literature review:

1. Jeyachandran et al. (2024) - ML techniques for real-time fraud detection
2. Sharma et al. (2025) - UPI-specific fraud detection
3. Achary & Shelke (2023) - Classical ML for banking fraud
4. Abakarim et al. (2018) - Deep learning for credit card fraud
5. Oza (2019) - Mobile payment fraud detection

## License

This project is for academic research purposes.

## Contact

For questions or collaboration:
- Shiva Gupta (23BCS10482)
- Uchit Yadav (23BCS10465)
- Priyanshu Saini (23BCS12371)
- Paramjeet Panchal (23BCS10104)

Supervised by: Er. Monika
Chandigarh University
