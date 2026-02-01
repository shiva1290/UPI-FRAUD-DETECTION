# UPI Fraud Detection System 🛡️

![Project Status](https://img.shields.io/badge/Status-Production%20Ready-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-Academic-lightgrey)

**A Next-Generation Hybrid Fraud Detection System combining Machine Learning and Large Language Models (LLM) for real-time UPI transaction security.**

---

## 🚀 Overview

This project implements a robust fraud detection pipeline designed for the Unified Payments Interface (UPI) ecosystem. Unlike traditional systems that rely solely on static rules or basic ML models, our system introduces a **Hybrid Architecture**:
1. **Fast ML Layer (Random Forest)**: Filters high-volume transactions in milliseconds (30ms latency).
2. **Cognitive LLM Layer (Llama 3.3 via Groq)**: Analyzes suspicious transactions to provide human-readable reasoning and advanced context awareness.

## ✨ Key Features

- **📊 Interactive Dashboard**: Real-time monitoring of transactions, fraud stats, and model performance.
- **🧠 Hybrid Intelligence**:
  - **Machine Learning**: Random Forest, XGBoost, LinearSVC (Accuracy > 99%).
  - **GenAI Reasoning**: Explains *why* a transaction is fraudulent using natural language.
- **⚡ Real-Time API**: RESTful endpoints for transaction scoring (Handle ~6000 RPM).
- **🛡️ Advanced Engineering**:
  - Device fingerprinting (location, device ID changes).
  - Behavioral profiling (velocity, beneficiary aging).
  - Network graphing (fan-in algorithms).
- **📈 Comprehensive Metrics**: Visualizes ROC-AUC, Recall, Precision, and Confusion Matrices.

## 🛠️ Technology Stack

- **Backend**: Python, Flask, Pandas, Scikit-learn, Joblib
- **Frontend**: HTML5, CSS3, JavaScript (Chart.js)
- **AI/LLM**: Groq API (Llama 3.3 70B), XGBoost
- **Data**: Synthetic UPI Transaction Dataset (Research Quality)

---

## 🚀 Getting Started

Follow these steps to set up the project locally.

### Prerequisites
- Python 3.9 or higher
- Git

### 1. Clone the Repository
```bash
git clone https://github.com/shiva1290/UPI-FRAUD-DETECTION.git
cd UPI-FRAUD-DETECTION
```

### 2. Install Dependencies
It's recommended to use a virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Setup Configuration
Create a `.env` file for API keys (optional, only for LLM features):
```bash
cp .env.example .env
# Edit .env and add your GROQ_API_KEY if you want LLM features
```

### 4. Run the System
You can start the full system (Dashboard + API) with a single command:
```bash
./start_dashboard.sh
```
*Alternatively, run manually:* `cd src && python app.py`

### 5. Access Dashboard
Open your browser and navigate to:
👉 **http://localhost:5000**

---

## 🧪 Testing & Verification

The project comes with a comprehensive test suite to ensure stability.

**Run Backend Tests:**
```bash
cd src
python test_api.py
```

**Train/Retrain Models:**
```bash
cd src
python train.py --with-llm  # Remove flag to skip LLM training
```

---

## 📚 Project Structure

```
.
├── data/               # Dataset storage
├── models/             # Trained ML models (.pkl)
├── results/            # Training metrics & visuals
├── src/                # Core source code
│   ├── app.py         # Flask API & Server
│   ├── models.py      # ML Model wrappers
│   ├── train.py       # Training pipeline
│   └── ...
├── web/                # Frontend assets
│   ├── static/        # CSS & JS
│   └── templates/     # HTML templates
└── ...
```

## 👨‍💻 Contributors

Developed at **Chandigarh University** under the supervision of **Er. Monika**.

- **Shiva Gupta** (23BCS10482)
- **Uchit Yadav** (23BCS10465)
- **Priyanshu Saini** (23BCS12371)
- **Paramjeet Panchal** (23BCS10104)

## 📄 License
This project is for academic research purposes.

---
*Built with ❤️ for a safer digital India.*
