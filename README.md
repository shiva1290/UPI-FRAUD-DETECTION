# UPI Fraud Detection System 🛡️

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)

**A hybrid fraud detection system that combines Machine Learning and Large Language Models (LLM) for real-time UPI transaction security.**

---

## Table of Contents

- [What This Project Is](#what-this-project-is)
- [What It Does](#what-it-does)
- [What It Follows](#what-it-follows)
- [Project Flow](#project-flow)
- [Technology Stack](#technology-stack)
- [Prerequisites](#prerequisites)
- [Setup: Create Project Directories](#setup-create-project-directories)
- [Configuration](#configuration)
- [Running the Project](#running-the-project)
- [Testing](#testing)
- [Project Structure](#project-structure)
- [Contributors](#contributors)
- [License](#license)

---

## What This Project Is

This project is a **hybrid fraud detection system** built for the **Unified Payments Interface (UPI)** ecosystem. It is designed for:

- **Academic research** and coursework (e.g., final-year or capstone projects)
- **Demonstrating** ML + LLM pipelines for financial security
- **Real-time scoring** of UPI-like transactions as fraud or legitimate
- **Explainable AI**: ML for speed, LLM for human-readable reasoning on suspicious cases

It is **not** a production banking system; it uses synthetic/research-quality data and is intended for learning and experimentation.

---

## What It Does

| Component | Purpose |
|----------|--------|
| **Data generation** | Creates synthetic UPI-style transactions (or uses optional real Kaggle data) with fraud patterns. |
| **Preprocessing** | Engineers features (velocity, device change, location, beneficiary fan-in, etc.) and scales/normalizes for ML. |
| **ML models** | Trains and compares Logistic Regression, Random Forest, XGBoost, SVM, Gradient Boosting; picks the best by F1. |
| **LLM layer** | Uses Groq (Llama 3.3) to analyze a sample of transactions and return fraud/legit + reasoning and risk factors. |
| **Flask API** | Serves predictions (`/api/predict`, `/api/predict_llm`), stats, model performance, and LLM samples. |
| **Dashboard** | Web UI for testing transactions, viewing model comparison, hourly fraud distribution, and LLM reasoning. |

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
│  4. Create .env (optional: GROQ_API_KEY for LLM)                         │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         TRAINING (train.py)                               │
├─────────────────────────────────────────────────────────────────────────┤
│  [1] Load or generate data (data/upi_transactions.csv)                    │
│  [2] Preprocess → save preprocessor (models/preprocessor.pkl)              │
│  [3] Train/validate/test 5 ML models → pick best by F1                     │
│  [4] Save best model (models/best_model_*.pkl + best_model_random_forest) │
│  [5] (Optional) Run LLM on sample → save (results/llm_predictions.csv)     │
│  [6] Save metrics & plots (results/model_performance.csv, *.png)           │
└─────────────────────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                         RUNTIME (app.py)                                   │
├─────────────────────────────────────────────────────────────────────────┤
│  • Load model + preprocessor from models/                                 │
│  • (If GROQ_API_KEY set) Initialize LLM detector                          │
│  • Serve dashboard (/) and API (/api/*)                                    │
│  • /api/predict, /api/predict_llm, /api/stats, /api/model_performance, …   │
└─────────────────────────────────────────────────────────────────────────┘
```

**Summary:** Setup → Train (data → preprocess → ML → optional LLM → save artifacts) → Run app (load artifacts, serve API and dashboard).

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

**Minimal `.env` for LLM:**

```env
GROQ_API_KEY=your_groq_api_key_here
```

Do **not** commit `.env` or real keys. The project never logs or prints the API key.

---

## Running the Project

Use the script for your operating system. Each script: creates a virtual environment if needed, installs dependencies, trains models if missing (with LLM when `GROQ_API_KEY` is in `.env`), then starts the dashboard.

### Option 1: One-command setup and run

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

**Command Prompt:** double-click `setup_and_run_windows.bat` or run:
```cmd
setup_and_run_windows.bat
```

**PowerShell:** (if execution policy allows)
```powershell
.\setup_and_run_windows.ps1
```
If you see an execution policy error, run once: `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`

**Generic Unix (Linux/macOS):** If you prefer the single script:

```bash
chmod +x setup_and_run.sh
./setup_and_run.sh
```

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
./start_dashboard.sh
# Or: cd src && python app.py
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

```
UPI-FRAUD-DETECTION/
├── data/                    # Datasets (e.g. upi_transactions.csv)
├── models/                  # Saved model and preprocessor (.pkl)
├── results/                 # model_performance.csv, plots, llm_predictions.csv
├── src/                     # Backend and training
│   ├── app.py              # Flask API and dashboard server
│   ├── config.py           # Configuration from .env
│   ├── train.py            # Full training pipeline (data → ML → optional LLM)
│   ├── preprocessor.py     # Feature engineering and scaling
│   ├── models.py           # ML model definitions and comparison
│   ├── llm_detector.py     # Groq-based LLM fraud detector
│   ├── data_generator.py   # Synthetic UPI data generation
│   └── test_api.py         # API tests
├── web/
│   ├── static/             # CSS, JS (e.g. dashboard.js)
│   └── templates/         # HTML (e.g. index.html)
├── .env.example            # Example environment variables
├── .env                     # Your config (do not commit)
├── requirements.txt         # Python dependencies
├── setup_and_run_linux.sh   # Setup + run (Linux)
├── setup_and_run_mac.sh     # Setup + run (macOS)
├── setup_and_run_windows.bat   # Setup + run (Windows CMD)
├── setup_and_run_windows.ps1   # Setup + run (Windows PowerShell)
├── setup_and_run.sh         # Setup + run (generic Unix)
├── start_dashboard.sh       # Start dashboard (Unix)
├── run.sh                   # Training with LLM (Unix)
├── demo_llm.sh              # Optional: CLI LLM demo (Unix; see README)
├── LLM_SETUP.md             # Optional: LLM setup (superseded by this README)
└── README.md                # This file
```

**Note:** LLM setup (Groq API key, `.env`) is in [Configuration](#configuration) above. To test LLM from the command line, run from project root: `cd src && source ../venv/bin/activate && python demo_llm.py` (or use the dashboard’s “Analyze with LLM”).

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
