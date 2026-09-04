# Machine Learning Sensor Fault Detection Project

An end-to-end Machine Learning pipeline for semiconductor wafer fault detection, built on the UCI SECOM dataset (1,567 samples, 397 sensor features, 93.4% Good / 6.6% Faulty).

---

## 🛠️ Architecture & Pipeline Overview

- **Data Ingestion**: Reads data from MongoDB Atlas with automatic fallback to a local cleaned dataset.
- **Data Transformation**: Median imputation (`SimpleImputer`) and robust scaling (`RobustScaler`) via a `scikit-learn` Pipeline.
- **Model Training**: Trains an `EasyEnsembleClassifier` (`imbalanced-learn`), chosen to handle the dataset's class imbalance.

---

## 🚀 Quickstart

### Installation & Environment Setup
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```
