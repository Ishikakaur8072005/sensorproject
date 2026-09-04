# Machine Learning Sensor Fault Detection Project

An end-to-end Machine Learning pipeline designed for semiconductor wafer fault detection on the UCI SECOM dataset (1,567 samples, 397 continuous sensor features, 93.4% Good / 6.6% Faulty imbalanced ratio).

---

## 🛠️ Architecture & Pipeline Overview

- **Data Ingestion**: Reads cleaned dataset from MongoDB Atlas with automatic fallback to `notebooks/wafer_secom_cleaned.csv`.
- **Data Transformation**: Applies median imputation (`SimpleImputer(strategy='median')`) and robust scaling (`RobustScaler`) via `scikit-learn` Pipeline.
- **Model Training**: Evaluates and trains `EasyEnsembleClassifier` from `imbalanced-learn`, optimized for fault recall and stability.
- **Web Interface & API**: Flask web app (`app.py`) exposing `/train` and `/predict` endpoints.

---

## 📊 Model Development & Selection

Due to the extreme class imbalance (1,463 Good Wafers vs 104 Faulty Wafers), traditional accuracy metric is misleading (a dummy 0-predicting model achieves 93.4% accuracy with 0% fault recall). Multiple imbalance-handling algorithms were evaluated using 5-Fold Stratified Cross-Validation (`StratifiedKFold(n_splits=5, shuffle=True, random_state=42)`):

| Model | Recall | False Alarm Rate | Why chosen / not chosen |
|---|---|---|---|
| **XGBoost (unweighted)** | 0.0% | 0.5% | 0% recall, always predicted majority class. |
| **SVC (`class_weight='balanced'`)** | 19.2% | 19.8% | Unstable across folds (recall std 0.38). |
| **BalancedRandomForestClassifier** | 23.1% | 7.1% | Solid conservative baseline, but lower recall than EasyEnsemble (23.1% vs 68.3%). |
| **RUSBoostClassifier** | 38.5% | 19.1% | Boosting-based alternative that achieved lower recall than EasyEnsemble (38.5% vs 68.3%). |
| **EasyEnsembleClassifier (Selected)** | **68.3%** | **32.6%** | **Selected:** Highest recall (68.3%) with strong cross-fold stability (F1 std 0.0166) under acceptable false alarm limits. |

---

## 🚀 Quickstart

### 1. Installation & Environment Setup
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Run Pipeline & Flask Web App
```bash
# Run training pipeline
python -m src.pipeline.train_pipeline

# Start Flask web server
python app.py
```
Access the application at `http://127.0.0.1:5000/`.