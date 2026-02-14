# Changelog – UPI Fraud Detection Project

Summary of changes made to get the project running end-to-end and improve usability, security, and documentation.

---

## 1. Making the Project Run End-to-End

- **Training → App model path**  
  The app expects `best_model_random_forest.pkl` by default, but training saved only `best_model_{name}.pkl` (e.g. xgboost). Training now also saves the best model as `best_model_random_forest.pkl` so the dashboard works without changing config.

- **Project directories**  
  Training now creates `models/`, `results/`, and `data/` if they don’t exist so the pipeline doesn’t fail on a fresh clone.

- **Config paths**  
  Model and preprocessor paths are resolved from the **project root** (not the current working directory), so the app finds `models/` and `results/` whether you run it from `src/` or the repo root.

- **App startup without models**  
  If model files are missing, the app still starts and logs a clear message (“Run training first…”) instead of exiting. ML endpoints return 503 until models exist.

- **Data generator without Kaggle**  
  If `data/kaggle_dataset_paths.txt` is missing, the data generator skips real data and uses synthetic data only, so training works without any Kaggle setup.

- **Dependency**  
  `imbalanced-learn` was added to `requirements.txt` because `models.py` uses it (SMOTE, etc.).

---

## 2. Dashboard & API Fixes

- **Feature importance**  
  The saved model is a wrapper object with an inner `.model`. The dashboard now reads feature importance from the inner model so the “Feature Importance” section works.

- **LLM samples section**  
  The CSV uses a column named `risk_factors`, but the dashboard expected `llm_risk_factors`. The API now exposes `llm_risk_factors` (mapped from `risk_factors`), and the frontend safely parses risk factors so the “LLM Predictions with Reasoning” section loads without errors.

- **Model names in performance comparison**  
  `model_performance.csv` has the model name in the first column (no header). The backend now detects that column and sets `model` in the JSON, and the frontend formats names (e.g. `random_forest` → “Random Forest”) so the table and chart no longer show “Unknown”.

- **ROC-AUC for LLM**  
  LLM metrics were not computing or saving ROC-AUC. Training now computes ROC-AUC from LLM confidence scores and adds it to the comparison so “LLM Groq” shows a ROC-AUC value instead of N/A.

- **Paths in app**  
  Paths for `model_performance.csv`, dataset, `llm_predictions.csv`, and web static/templates are resolved from the project root so the app works regardless of the current working directory.

---

## 3. LLM & Training Fixes

- **LLM column name bug**  
  `analyze_sample_predictions()` expected a column `llm_is_fraud`, but `predict_batch()` writes `llm_prediction`. The code was updated to use `llm_prediction` and to handle missing or failed LLM results so training doesn’t crash.

- **API key handling**  
  The Groq API key read from `.env` could include quotes or whitespace, causing 401 errors. The key is now stripped of quotes and whitespace when loaded in both `train.py` and `llm_detector.py`. `.env` is also loaded with `python-dotenv` at the start of `train.py` so the key is available and parsed correctly.

---

## 4. Security

- **No API key in logs or output**  
  Training and the app no longer print or log any part of the API key. Messages were changed from “API Key detected: gsk_xx...yyyy” to “API key detected.” so the key is never exposed in terminal or log files.

---

## 5. Setup & Run Scripts

- **`setup_and_run.sh`**  
  New one-command script that: creates a virtual environment if missing, installs dependencies, trains models (and runs LLM training if `GROQ_API_KEY` is set in `.env`), then starts the dashboard.

- **Optional LLM in setup**  
  If `.env` exists and contains a non-empty `GROQ_API_KEY`, `setup_and_run.sh` runs `train.py --with-llm`; otherwise it runs `train.py` (ML only).

---

## 6. Documentation & License

- **README.md**  
  Rewritten into a full guide: what the project is, what it does, what it follows, project flow (with a flow diagram), tech stack, prerequisites, **commands to create project directories on Linux, macOS, Windows PowerShell, and Windows CMD**, configuration (`.env` variables table and example), running the project on all platforms, testing, and project structure.

- **LICENSE**  
  Added an MIT License file stating that the project is owned by the UPI Fraud Detection Team (Chandigarh University), is open source, and listing the contributors. Copyright and permission/disclaimer text follow the standard MIT template.

- **`.env.example`**  
  Already present; documented in the README as the template for `GROQ_API_KEY` and other optional settings.

---
securtiy
## Files Touched (Summary)securtiy

| Area            | Files |
|-----------------|--------|
| Training        | `src/train.py`, `src/data_generator.py` |
| App & config    | `src/app.py`, `src/config.py` |
| LLM             | `src/llm_detector.py` |
| Frontend        | `web/static/js/dashboard.js` |
| Dependencies    | `requirements.txt` |
| Scripts         | `setup_and_run.sh` (new) |
| Docs & legal    | `README.md`, `LICENSE` (new), `CHANGELOG.md` (this file) |

---

You can use this as a “Description of changes” for submissions, README sections, or version notes.
