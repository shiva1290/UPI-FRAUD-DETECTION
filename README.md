# UPI Fraud Risk Assessment Framework 🛡️

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**A hybrid ML + explainable reasoning system for UPI transaction fraud detection that combines fast machine learning with human-readable LLM explanations.**

---

## Key Contributions

> **For reviewers:** Contribution is spread across sections; this block summarizes the novelty at a glance.

| Contribution | Description |
|--------------|-------------|
| **Risk-based decision layer** | Introduced a decision layer *between* ML prediction and explanation. Configurable thresholds (allow/review/block) replace raw binary labels and drive when LLM is invoked. |
| **Conditional LLM execution** | LLM is triggered only for medium/high-risk cases, not for every transaction. Reduces computational overhead and API cost. *The LLM is not used for fraud prediction but only for generating explanations for high-risk transactions.* |
| **Hybrid architecture** | ML handles real-time scoring; LLM adds interpretability on demand. Improves explainability without affecting inference speed. |

---

## Table of Contents

- [Key Contributions](#key-contributions)
- [Problem Statement](#problem-statement)
- [Contribution](#contribution)
- [What This Project Is](#what-this-project-is)
- [What It Does](#what-it-does)
- [ML Metrics & Interpretation](#ml-metrics--interpretation)
- [Feature importance interpretation](#feature-importance-interpretation)
- [Risk Assessment Framework](#risk-assessment-framework)
- [Architecture](#architecture)
- [Limitations](#limitations)
- [Future Work](#future-work)
- [What It Follows](#what-it-follows)
- [Project Flow](#project-flow)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Setup: Create Project Directories](#setup-create-project-directories)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Changelog Summary](#changelog-summary)
- [Contributors](#contributors)
- [License](#license)

---

## Problem Statement

Traditional fraud detection systems face a critical **explainability gap**: ML models (e.g., Random Forest, XGBoost) achieve high accuracy but output only binary labels or probability scores without *why* a transaction was flagged. Regulators, auditors, and users need interpretable reasoning for compliance, dispute resolution, and trust. Rule-based systems, while interpretable, lack the adaptability and performance of modern ML. This project addresses the **explainability gap** by bridging high-performing ML with human-readable reasoning.

---

## Contribution

This project contributes a **hybrid ML + explainable reasoning system**:

- **Fast ML layer**: Random Forest (or best F1 model) produces fraud probability and risk scores in milliseconds.
- **Explainable reasoning layer**: Rule-based explanations for all transactions; optional LLM (Groq/Llama 3.3) for natural-language reasoning on medium/high-risk cases.
- **Risk assessment framework**: Configurable thresholds map scores to actions (allow/review/block), not just fraud/legitimate.
- **Dual latency visibility**: ML vs LLM latency comparison for transparency and optimization.

The system is designed for academic research and demonstrates how to close the explainability gap in financial fraud detection.

---

## What This Project Is

This project is a **risk assessment framework** for UPI transaction fraud detection, not a binary classifier. It produces **risk scores (0–100)**, **risk levels (Low/Medium/High)**, and **system actions (allow/review/block)** instead of a simple fraud/legitimate label. It is built for:

- **Academic research** and coursework (e.g., final-year or capstone projects)
- **Demonstrating** ML + LLM pipelines for financial security
- **Risk-based decisions**: configurable thresholds drive allow/review/block actions
- **Explainable AI**: rule-based explanations for all; optional LLM reasoning for medium/high risk

It is **not** a production banking system; it uses synthetic/research-quality data and is intended for learning and experimentation.

---

## What It Does

| Component | Purpose |
|----------|--------|
| **Feature engineering** | Prepares raw transactions for ML (velocity, device change, location, beneficiary fan-in, etc.). |
| **ML prediction** | Produces fraud probability (0–1) from trained model. |
| **Risk engine** | Converts probability → risk score (0–100) → risk level (Low/Medium/High) → action (allow/review/block). |
| **LLM explanation** | Rule-based explanations for all; optional Groq (Llama 3.3) for human-readable reasoning on medium/high risk. |
| **Flask API** | Serves risk assessments (`/api/predict`, `/api/predict_llm`), stats, model performance, confusion matrix. |
| **Dashboard** | Web UI with risk gauge, confusion matrix heatmap, metrics tooltips, latency comparison, and optional LLM analysis per transaction. |

---

## ML Metrics & Interpretation

The dashboard explains key metrics via hover tooltips. Below is the full theory (from `metrics_analysis.py`) that backs those tooltips.

### Why These Metrics Matter

**Imbalanced datasets.** Fraud detection datasets typically have far fewer fraudulent transactions (often 1–5%) than legitimate ones. In such imbalanced settings, *accuracy is misleading*: a model that predicts "legitimate" for everything can achieve 95%+ accuracy while missing all fraud. We prioritize Precision, Recall, and F1-score instead of accuracy.

**Why recall matters.** Recall (True Positives / (True Positives + False Negatives)) measures what fraction of actual frauds we catch. In fraud detection, missing a fraudulent transaction (false negative) is often costlier than flagging a legitimate one (false positive). Low recall means fraud slips through; high recall reduces financial loss and customer trust issues. We balance recall with precision to avoid too many false alarms.

**False positives vs false negatives.** *False positive*: A legitimate transaction wrongly flagged as fraud. Impact: customer friction, blocked payments, support load, lost revenue. *False negative*: A fraudulent transaction wrongly allowed. Impact: direct financial loss, chargebacks, regulatory risk, reputational damage. For UPI fraud, false negatives are typically more costly. We use F1-score to balance both, and ROC-AUC to assess ranking quality across thresholds.

**Why moderate recall (~48%) is acceptable.** Moderate recall reduces false alarms: chasing very high recall often means lowering the decision threshold, which sharply increases false positives. Too many false alarms overload support, frustrate customers, and block legitimate payments. A ~48% recall balances catching fraud with manageable alert volume. *Higher recall increases false positives significantly; therefore a balanced threshold was selected to avoid unnecessary transaction blocking.*

**Threshold trade-off.** Raising the decision threshold increases precision and lowers recall; lowering it does the opposite.

### Metric definitions

| Metric | Definition |
|--------|------------|
| **Precision** | Of transactions flagged as fraud, how many were actually fraud? |
| **Recall** | Of all actual frauds, how many did we catch? Moderate recall (~48%) limits false alarms. |
| **F1-score** | Harmonic mean of precision and recall; balances both. |
| **ROC-AUC** | Model's *discrimination ability* across thresholds—how well it ranks fraud vs legitimate—not an accuracy indicator. |
| **Accuracy** | Fraction of correct predictions. Less reliable for imbalanced fraud datasets. |
| **Confusion matrix** | TN, FP, FN, TP heatmap for the selected model (saved during training). |

### Selected model: Random Forest

Random Forest is the final selected model based on balanced Precision and Recall. It provides good F1-score and ROC-AUC, handles imbalanced data well, and supports feature importance for explainability. The LLM module adds human-readable reasoning for medium/high-risk cases; it is not a classifier.

### ML classifiers vs LLM explanation module

| Aspect | ML classifiers (e.g., Random Forest) | LLM explanation module (Groq) |
|--------|-------------------------------------|-------------------------------|
| **Strengths** | Fast inference (~30ms), consistent behavior, no API dependency, scales to high throughput, classifies fraud and produces risk scores | Natural-language explanations, aligns with model features, useful for compliance and dispute resolution, invoked only for medium/high risk (cost-efficient), adds interpretability |
| **Limitations** | No human-readable reasoning, black-box, fixed feature set | Slower (~2–5s per call), requires API key and incurs cost, non-deterministic |

**Recommendation.** Use ML classifiers for real-time fraud classification. Use the LLM explanation module to generate human-readable reasoning for medium/high-risk cases. The LLM is not a classifier; it explains why a transaction was flagged.

**Clarification.** The LLM is not used for fraud prediction but only for generating explanations for high-risk transactions.

### Feature importance interpretation

The dashboard plots top contributing fraud features. In our model, *transaction velocity* and *amount deviation* were the most influential features in fraud prediction, followed by device change and beneficiary-related signals. This reflects real-world patterns: rapid or unusual transaction patterns and large deviations from typical amounts are strong fraud indicators.

---

## Risk Assessment Framework

The system is a **risk assessment framework**, not a binary classifier:

| Concept | Description |
|---------|-------------|
| **Risk score** | 0–100, derived from ML fraud probability. Higher = more likely fraud. |
| **Thresholds** | Configurable via `.env`: `RISK_LOW_THRESHOLD` (default 30), `RISK_MEDIUM_THRESHOLD` (default 70). |
| **Risk levels** | Low (score &lt; low_threshold), Medium (low ≤ score &lt; medium), High (score ≥ medium). |
| **Actions** | Low → allow; Medium → review; High → block. |
| **LLM** | User can optionally request LLM explanation for medium/high risk (button in UI). |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         RISK ASSESSMENT PIPELINE (Runtime)                        │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                   │
│   Raw Transaction                                                                │
│         │                                                                         │
│         ▼                                                                         │
│   ┌──────────────────┐                                                           │
│   │ Feature          │  Clean, engineer, encode, scale                            │
│   │ Engineering      │  (preprocessor, feature_engineering.py)                    │
│   └────────┬─────────┘                                                           │
│            │                                                                      │
│            ▼                                                                      │
│   ┌──────────────────┐                                                           │
│   │ ML Prediction    │  predict_proba → fraud probability (0–1)                   │
│   │ (ml_prediction)  │                                                           │
│   └────────┬─────────┘                                                           │
│            │                                                                      │
│            ▼                                                                      │
│   ┌──────────────────┐     ┌─────────────────────────────┐                       │
│   │ Risk Engine      │     │ Threshold Layer             │                       │
│   │ (risk_engine)    │────▶│ Low < 30 | Med 30–70 | High │                       │
│   │                  │     │ (configurable via .env)     │                       │
│   └────────┬─────────┘     └─────────────────────────────┘                       │
│            │                                                                      │
│            ▼                                                                      │
│   ┌──────────────────┐                                                           │
│   │ Decision Layer   │  allow | review | block                                    │
│   │ (inside engine)  │                                                           │
│   └────────┬─────────┘                                                           │
│            │                                                                      │
│            ▼                                                                      │
│   ┌──────────────────┐                                                           │
│   │ LLM Explanation  │  Rule-based always; LLM on user request (medium/high)      │
│   │ (llm_explanation)│                                                           │
│   └──────────────────┘                                                           │
│                                                                                   │
└─────────────────────────────────────────────────────────────────────────────────┘
```

**Module layout (SOLID):**

| Module | Responsibility |
|--------|----------------|
| **feature_engineering** | Prepares raw transaction data for ML (clean, engineer, encode, scale) |
| **ml_prediction** | Produces fraud probability (0–1) from the trained model |
| **risk_engine** | Converts probability → risk score → risk level → decision |
| **llm_explanation** | Generates rule-based or LLM-based explanations |
| **confusion_matrix_store** | Persists and loads confusion matrix for the selected model (SRP) |
| **metrics_analysis** | Provides explanatory text for precision, recall, ROC-AUC, and threshold trade-offs |

**Threshold configuration:**

| Threshold | Default | Meaning |
|-----------|---------|---------|
| `RISK_LOW_THRESHOLD` | 30 | Score &lt; 30 = Low risk → allow |
| `RISK_MEDIUM_THRESHOLD` | 70 | 30 ≤ score &lt; 70 = Medium → review; score ≥ 70 = High → block |

---

## Limitations

- **Synthetic data**: The system uses synthetic or research-quality UPI datasets. Real production data may have different distributions, attack patterns, and regulatory constraints. Results are not directly applicable to live banking systems without validation on real data.

- **Threshold dependency**: Risk levels and actions (allow/review/block) depend on configurable thresholds. Suboptimal thresholds can over-block legitimate transactions or under-flag fraud. Thresholds should be tuned per deployment context.

- **LLM latency and cost**: LLM reasoning adds hundreds of milliseconds per request and consumes API credits. It is intended as an optional, on-demand layer for medium/high-risk cases, not for every transaction.

- **Moderate recall by design**: The selected model targets ~48% recall to limit false alarms. Higher recall increases false positives significantly; therefore a balanced threshold was selected to avoid unnecessary transaction blocking. See dashboard tooltips for details.

- **Academic scope**: This project is for learning and demonstration. It is not audited for production use, regulatory compliance, or security hardening.

---

## Future Work

- **Real-time streaming**: Extend the pipeline to consume transaction streams (e.g., Kafka, event queues) for continuous scoring and alerting in near real time.

- **Graph-based fraud detection**: Model beneficiary networks, device graphs, and money-flow paths to detect organized fraud rings and multi-hop laundering patterns.

- **Adaptive thresholds**: Learn or recommend thresholds from feedback (chargebacks, false positives) to reduce manual tuning.

- **Production hardening**: Add audit logging, role-based access, and integration with banking middleware and regulatory reporting.

---

## What It Follows

- **Hybrid architecture**: Fast ML filter + optional LLM for explainability.
- **Standard ML workflow**: Load data → preprocess → train/validate/test → persist best model and preprocessor.
- **REST API design**: JSON in/out, clear routes, rate limiting, optional API-key protection.
- **Configuration via environment**: `.env` for secrets and options; no hardcoded keys.
- **Reproducibility**: Fixed random seeds, versioned dependencies in `requirements.txt`.

---

## Project Flow

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         SETUP (one-time)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  1. Clone repo → 2. Create venv → 3. pip install -r requirements.txt    │
│  4. Create .env (optional: GROQ_API_KEY, RISK_LOW_THRESHOLD, etc.)       │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING (train.py)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  [1] Load or generate data (data/upi_transactions.csv)                   │
│  [2] Feature engineering + preprocess → save preprocessor                │
│  [3] Train/validate/test 5 ML models → pick best by F1                   │
│  [4] Save best model (models/best_model_*.pkl + best_model_random_forest)│
│  [5] Save confusion matrix (results/confusion_matrix.json, *.png)        │
│  [6] (Optional) Run LLM on sample → save (results/llm_predictions.csv)   │
│  [7] Save metrics & plots (results/model_performance.csv, *.png)         │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RUNTIME (app.py)                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  • Load model + preprocessor; init risk engine (thresholds from .env)    │
│  • (If GROQ_API_KEY set) Initialize LLM detector                         │
│  • Pipeline: feature_engineering → ml_prediction → risk_engine → explain │
│  • Serve dashboard (/) and API (/api/predict, /api/predict_llm, …)       │
└─────────────────────────────────────────────────────────────────────────┘
```

**Summary:** Setup → Train (data → feature engineering → ML → optional LLM → save) → Run (risk pipeline: features → ML prob → risk score → threshold → decision → explanation).

---

## Technology Stack

| Layer | Technologies |
|-------|--------------|
| **Backend** | Python 3.9+, Flask, Pandas, NumPy, Scikit-learn, XGBoost, Joblib |
| **Frontend** | HTML5, CSS3, JavaScript, Chart.js |
| **AI/LLM** | Groq API (Llama 3.3 70B) |
| **Data** | Synthetic UPI dataset; optional Kaggle integration |
| **Config** | python-dotenv, `.env` |

---

## Prerequisites

- **Python** 3.9 or higher  
- **Git**  
- (Optional) **Groq API key** for LLM features: [Groq Console](https://console.groq.com/keys)

---

## Setup: Create Project Directories

After cloning, create the directories and virtual environment. Use the commands for your OS.

### Linux / macOS (bash/zsh)

```bash
# Clone
git clone https://github.com/shiva1290/UPI-FRAUD-DETECTION.git
cd UPI-FRAUD-DETECTION

# Create directories (if not already present)
mkdir -p data models results web/static web/templates src logs

# Virtual environment
python3 -m venv venv
source venv/bin/activate

# Dependencies
pip install -r requirements.txt
```

### Windows (PowerShell)

```powershell
# Clone
git clone https://github.com/shiva1290/UPI-FRAUD-DETECTION.git
cd UPI-FRAUD-DETECTION

# Create directories
New-Item -ItemType Directory -Force -Path data, models, results, web\static, web\templates, src, logs

# Virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Dependencies
pip install -r requirements.txt
```

### Windows (Command Prompt)

```cmd
git clone https://github.com/shiva1290/UPI-FRAUD-DETECTION.git
cd UPI-FRAUD-DETECTION

mkdir data 2>nul & mkdir models 2>nul & mkdir results 2>nul
mkdir web\static 2>nul & mkdir web\templates 2>nul & mkdir src 2>nul & mkdir logs 2>nul

python -m venv venv
venv\Scripts\activate.bat

pip install -r requirements.txt
```

**Note:** The repo may already include `data/`, `models/`, `results/`, `web/`, `src/`. The commands above ensure they exist; creating them again is safe.

---

## Configuration

Configuration is done via a **`.env`** file in the project root. Copy the example and edit as needed:

```bash
# Linux/macOS
cp .env.example .env

# Windows (PowerShell)
Copy-Item .env.example .env
```

### Environment variables

| Variable | Required | Description |
|---------|----------|-------------|
| `GROQ_API_KEY` | For LLM only | Groq API key for Llama 3.3. Get from [Groq Console](https://console.groq.com/keys). |
| `LLM_ENABLED` | No | Set to `false` to disable LLM even if key is set. Default: `true` when key is set. |
| `LLM_MODEL` | No | Groq model name. Default: `llama-3.3-70b-versatile`. |
| `API_HOST` | No | Bind address. Default: `0.0.0.0`. |
| `API_PORT` | No | Port. Default: `5000`. |
| `DEBUG` | No | Flask debug. Default: `False`. |
| `RATE_LIMIT_PER_MINUTE` | No | Max requests per minute per IP. Default: `60`. |
| `MODEL_PATH` | No | Path to best ML model. Default: `models/best_model_random_forest.pkl`. |
| `PREPROCESSOR_PATH` | No | Path to preprocessor. Default: `models/preprocessor.pkl`. |
| `RISK_LOW_THRESHOLD` | No | Score below this = Low risk. Default: `30`. |
| `RISK_MEDIUM_THRESHOLD` | No | Score below this = Medium; above = High. Default: `70`. |

**Minimal `.env` for LLM:**

```env
GROQ_API_KEY=your_groq_api_key_here
```

Do **not** commit `.env` or real keys. The project never logs or prints the API key.

---

## Running the Project

Use the script for your operating system. Each script: creates a virtual environment if needed, installs dependencies, trains models if missing (with LLM when `GROQ_API_KEY` is in `.env`), then starts the dashboard.

### Option 1: One-command setup and run

Setup scripts are at **project root** (not in `bin/`). Run from project root:

**Linux:**

```bash
chmod +x setup_and_run_linux.sh
./setup_and_run_linux.sh
```

**macOS:**

```bash
chmod +x setup_and_run_mac.sh
./setup_and_run_mac.sh
```

**Windows:**

**Command Prompt:**
```cmd
setup_and_run_windows.bat
```

**PowerShell:** (if execution policy allows)
```powershell
.\setup_and_run_windows.ps1
```
If you see an execution policy error, run once: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

This will: create venv if missing, install dependencies, train models (with LLM if `GROQ_API_KEY` is in `.env`), then start the dashboard.

### Option 2: Train once, then start dashboard

**Linux / macOS:**

```bash
# Activate venv
source venv/bin/activate

# Train (ML only; no API key needed)
cd src && python train.py && cd ..

# Or train with LLM (requires GROQ_API_KEY in .env)
cd src && python train.py --with-llm && cd ..

# Start dashboard
cd src && python app.py
```

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
cd src; python train.py; cd ..
# Optional: python train.py --with-llm
cd src; python app.py
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
cd src && python train.py && cd ..
cd src && python app.py
```

### Access

- **Dashboard:** [http://localhost:5000](http://localhost:5000)  
- **Health:** [http://localhost:5000/health](http://localhost:5000/health)  
- **Confusion matrix API:** [http://localhost:5000/api/confusion_matrix](http://localhost:5000/api/confusion_matrix) (JSON: `{matrix, model}`)

---

## Testing

With the app running in another terminal:

```bash
# Linux/macOS
source venv/bin/activate
cd src && python test_api.py

# Windows
venv\Scripts\activate
cd src
python test_api.py
```

---

## Project Structure

Structure below reflects **tracked files**. `bin/`, `logs/`, `venv/` and generated files are in [.gitignore](.gitignore).

```
UPI-FRAUD-DETECTION/
├── data/                    # Datasets (data/*.csv gitignored; .gitkeep, kaggle_dataset_paths.txt tracked)
├── models/                  # Saved model and preprocessor (models/*.pkl gitignored)
├── results/                 # Metrics and plots (results/*.csv, *.png gitignored; .gitkeep tracked)
├── src/                     # Backend and training
│   ├── app.py              # Flask API and dashboard server
│   ├── config.py           # Configuration from .env
│   ├── train.py            # Full training pipeline (data → ML → optional LLM)
│   ├── preprocessor.py     # Feature engineering and scaling
│   ├── feature_engineering.py  # Prepares transactions for ML
│   ├── ml_prediction.py    # Fraud probability prediction
│   ├── risk_engine.py      # Risk scoring + thresholds + decision layer
│   ├── llm_explanation.py  # Rule-based or LLM explanations
│   ├── models.py           # ML model definitions and comparison
│   ├── llm_detector.py     # Groq-based LLM fraud detector
│   ├── data_generator.py
│   ├── download_datasets.py
│   ├── demo_llm.py
│   ├── abstractions.py
│   ├── decision_layer.py
│   ├── risk_scoring.py
│   ├── llm_client.py
│   ├── llm_fraud_detector.py
│   ├── explanation_generator.py
│   ├── feature_importance_store.py
│   ├── confusion_matrix_store.py
│   ├── prediction_service.py
│   ├── prediction_store.py
│   ├── prediction_logger.py
│   ├── transaction_validator.py
│   ├── model_registry.py
│   ├── model_visualizer.py
│   ├── metrics_analysis.py
│   └── test_api.py
├── web/
│   ├── static/             # CSS, JS (e.g. dashboard.js)
│   └── templates/          # HTML (e.g. index.html)
├── setup_and_run_linux.sh
├── setup_and_run_mac.sh
├── setup_and_run_windows.bat
├── setup_and_run_windows.ps1
├── .env.example
├── .env                    # Your config (do not commit)
├── requirements.txt
├── LICENSE
└── README.md
```

**Ignored by .gitignore:** `bin/`, `logs/`, `venv/`, `data/*.csv`, `models/*.pkl`, `results/*.csv`, `results/*.png`, `results/confusion_matrix.json`, `*.log`.

**Note:** LLM setup (Groq API key, `.env`) is in [Configuration](#configuration) above. To test LLM: `cd src && python demo_llm.py` (or use the dashboard’s “Analyze with LLM”).

---

## Changelog Summary

- **Confusion matrix**: Visualization in dashboard for selected model; `ConfusionMatrixStore` saves/loads `results/confusion_matrix.json`; `/api/confusion_matrix` endpoint.
- **Metrics explanations**: Tooltips for moderate recall (~48%) acceptable to avoid false alarms; one-line threshold trade-off (precision vs recall); ROC-AUC described as discrimination ability, not accuracy.
- Training now saves `best_model_random_forest.pkl` and confusion matrix JSON for dashboard compatibility.
- Paths resolved from project root; app starts even if models are missing (503 until trained).
- Feature importance, LLM samples, and model names fixed for frontend.
- ROC-AUC added for LLM metrics; prediction logger and latency measurement added.
- Risk gauge, latency comparison, and prediction logs for fraud pattern analysis.
- API key handling and security improvements; setup scripts for Linux, macOS, Windows.

*(Full changelog available in `bin/CHANGELOG.md` when present locally; `bin/` is gitignored.)*

---

## Contributors

Developed at **Chandigarh University** under the supervision of **Er. Monika**.

- **Shiva Gupta** (23BCS10482)
- **Uchit Yadav** (23BCS10465)
- **Priyanshu Saini** (23BCS12371)
- **Paramjeet Panchal** (23BCS10104)

---

## License

This project is **open source** and **owned by the contributing team**. It is released under the **MIT License**.

You may use, copy, modify, merge, publish, distribute, sublicense, and sell copies of the software, subject to the conditions in the [LICENSE](LICENSE) file. The license requires preservation of the copyright and license notice. The authors and the team are not liable for any claim or damages arising from the use of the software.

See the [LICENSE](LICENSE) file in the repository for the full text.
