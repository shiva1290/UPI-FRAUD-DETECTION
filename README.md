# UPI Fraud Risk Assessment Framework 🛡️

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**A hybrid ML + explainable reasoning system for UPI transaction fraud detection that combines fast machine learning with human-readable LLM explanations.**

---

## Key Contributions

| Contribution | Description |
|--------------|-------------|
| **Risk-based decision layer** | Introduced a decision layer *between* ML prediction and explanation. Configurable thresholds (allow/review/block) replace raw binary labels. |
| **Conditional LLM execution** | LLM is triggered only for medium/high-risk cases, reducing computational overhead. *The LLM is not used for fraud prediction but only for generating explanations.* |
| **Hybrid architecture** | ML handles real-time scoring; LLM adds interpretability on demand without affecting inference speed. |
| **PR & cost-sensitive evaluation** | Precision–Recall curves, PR-AUC, cost-sensitive evaluation (FN cost > FP cost), expected financial loss tables, and cost–vs–threshold curves. |
| **LLM vs SHAP comparison** | Side-by-side comparison of SHAP (quantitative) and LLM (natural language) explanations with alignment scoring. |
| **Concept drift simulation** | Simulates performance degradation over time as data distributions shift. |
| **Explanation quality rating** | Human evaluation system (1-5 scale) for rating explanation quality. |
| **PR tradeoff analysis** | Precision-Recall tradeoff comparison between Random Forest and XGBoost across multiple thresholds. |
| **External dataset benchmarks** | Automated benchmarking on Kaggle datasets (ULB Credit Card, PaySim) with dedicated training script and logging. |

---

## What This Project Is

A **risk assessment framework** for UPI transaction fraud detection that produces **risk scores (0–100)**, **risk levels (Low/Medium/High)**, and **system actions (allow/review/block)**. Built for academic research and demonstrating ML + LLM pipelines for financial security.

---

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
# Clone repository
git clone <repository-url>
cd UPI-FRAUD-DETECTION

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Optional: Set up LLM (requires Groq API key)
cp .env.example .env
# Edit .env and add: GROQ_API_KEY=your_key_here
```

### Running

**Option 1: One-command setup (Linux/macOS)**
```bash
chmod +x setup_and_run_linux.sh
./setup_and_run_linux.sh
```

**Option 2: Manual**
```bash
# Train models
cd src
python train.py

# Start dashboard
python app.py
```

**Access:** [http://localhost:5000](http://localhost:5000)

---

## Features

### Core Features
- **ML Models**: Random Forest, XGBoost, Logistic Regression, SVM, Gradient Boosting
- **Risk Engine**: Probability → Risk Score → Risk Level → Action (allow/review/block)
- **LLM Explanations**: Optional Groq (Llama 3.3) for natural-language reasoning
- **Dashboard**: Web UI with visualizations, metrics, and transaction testing

### Advanced Features
- **LLM vs SHAP Comparison**: Side-by-side explanation comparison with alignment scores
- **Concept Drift Simulation**: Performance degradation tracking over time periods
- **Explanation Quality Rating**: Human rating system (1-5 scale) for explanations
- **PR Tradeoff Analysis**: RF vs XGBoost comparison across multiple thresholds
- **External Benchmarks**: Automated benchmarking on Kaggle datasets (ULB Credit Card, PaySim)

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/predict` | POST | ML-based fraud risk assessment |
| `/api/predict_llm` | POST | LLM explanation for transaction |
| `/api/model_performance` | GET | Model performance metrics |
| `/api/pr_curves` | GET | Precision-Recall curves for all models |
| `/api/cost_curve` | GET | Cost vs threshold curve |
| `/api/concept_drift` | GET | Concept drift simulation results |
| `/api/pr_tradeoff` | GET | PR tradeoff comparison (RF vs XGBoost) |
| `/api/shap_global` | GET | Global SHAP feature importance |
| `/api/explanation_comparison` | POST | Compare SHAP vs LLM explanations |
| `/api/explanation_rating` | POST | Submit explanation quality rating |
| `/api/external_benchmark` | GET | External dataset benchmark results |

---

## External Dataset Benchmarks

### Setup

1. **Download datasets from Kaggle:**
   - ULB Credit Card: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
   - PaySim: https://www.kaggle.com/datasets/ealaxi/paysim1

2. **Place files in `data/` directory:**
   ```bash
   cp /path/to/creditcard.csv data/creditcard.csv
   cp /path/to/paysim.csv data/paysim.csv  # or PS_*.csv
   ```

3. **Run benchmarks:**
   ```bash
   # Dedicated script (recommended for large datasets)
   ./run_external_benchmarks.sh
   
   # Or as part of main training
   cd src && python train.py
   ```

**Results:** Saved to `results/external_benchmark.csv` and displayed in dashboard.

---

## Project Structure

```
UPI-FRAUD-DETECTION/
├── data/                    # Datasets (gitignored)
├── models/                  # Saved models (gitignored)
├── results/                 # Metrics and plots (gitignored)
├── logs/                    # Training logs (gitignored)
├── src/                     # Backend code
│   ├── app.py              # Flask API and dashboard
│   ├── train.py            # Main training pipeline
│   ├── train_external_benchmarks.py  # External dataset training
│   ├── models.py           # ML model definitions
│   ├── preprocessor.py      # Feature engineering
│   ├── risk_engine.py      # Risk scoring and thresholds
│   ├── concept_drift_simulator.py
│   ├── pr_tradeoff_analyzer.py
│   ├── explanation_comparison.py
│   ├── explanation_rating_store.py
│   ├── shap_explainer.py
│   ├── external_benchmark.py
│   └── ...                 # Other modules
├── web/                     # Frontend
│   ├── static/             # CSS, JS
│   └── templates/          # HTML
├── requirements.txt
├── .env.example
└── README.md
```

---

## Configuration

Create `.env` file (copy from `.env.example`):

```env
# Optional: LLM API Key
GROQ_API_KEY=your_groq_api_key_here

# Optional: Risk Thresholds
RISK_LOW_THRESHOLD=30
RISK_MEDIUM_THRESHOLD=70

# Optional: API Settings
API_HOST=0.0.0.0
API_PORT=5000
DEBUG=False
```

---

## Technology Stack

- **Backend**: Flask, Python 3.9+
- **ML**: scikit-learn, XGBoost, imbalanced-learn
- **Explainability**: SHAP
- **LLM**: Groq API (Llama 3.3)
- **Frontend**: HTML5, CSS3, JavaScript, Chart.js
- **Data**: Pandas, NumPy

---

## Key Metrics

- **Precision**: Fraction of flagged transactions that are actually fraud
- **Recall**: Fraction of fraud cases that are detected
- **F1-Score**: Harmonic mean of precision and recall
- **PR-AUC**: Better than ROC-AUC for imbalanced fraud data
- **ROC-AUC**: Discrimination ability across thresholds
- **Cost-Sensitive**: FN cost (₹10,000) > FP cost (₹1,000)

**Selected Model**: Random Forest (best F1-score)
- Moderate recall (~48%) to avoid excessive false alarms
- Higher recall increases false positives significantly

---

## Architecture

**Pipeline:** Raw Transaction → Feature Engineering → ML Prediction → Risk Engine → Decision (allow/review/block) → Explanation

**SOLID Principles:**
- Single Responsibility: Each module has one clear purpose
- Open/Closed: Extensible without modifying existing code
- Dependency Inversion: Uses abstractions

---

## Contributors

Developed at **Chandigarh University** under the supervision of **Er. Monika**.

- **Shiva Gupta** (23BCS10482)
- **Uchit Yadav** (23BCS10465)
- **Priyanshu Saini** (23BCS12371)
- **Paramjeet Panchal** (23BCS10104)

---

## License

MIT License - See [LICENSE](LICENSE) file for details.
