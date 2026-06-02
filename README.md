# Credit Scoring Model — NTACCUL Microfinance (Cameroon)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Production--Ready-brightgreen)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

End-to-end **credit scoring model** developed for **NTACCUL**, a microfinance institution in Cameroon.  
The goal is to predict the probability of loan default on a portfolio of **255,000 loan files** (amounts in FCFA), enabling proactive risk management and optimised lending policy decisions.

The project covers the full modelling lifecycle:

```
Data Ingestion → Exploratory Analysis → Feature Engineering
→ Model Training & Benchmarking → Calibration & Scorecard
→ Production Deployment → Automated Monitoring & Alerting
```

---

## Key Results

| Metric | Baseline Model | Final Model |
|--------|---------------|-------------|
| Default Detection Rate | 1× (reference) | **+910%** |
| At-Risk Cases Identified | — | **3,030 / 51,000** |
| AUC (reference cohort) | — | **73.9%** |
| AUC stability (2020–2024) | — | [72.6% – 73.5%] |
| Monitoring | None | Automated alerts |

---

## Project Structure

```
ntaccul_credit_scoring/
│
├── data/
│   ├── raw/                  # Original loan files (not versioned)
│   └── processed/            # Cleaned & feature-engineered datasets
│
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_training.ipynb
│   ├── 04_calibration_scorecard.ipynb
│   └── 05_monitoring.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py    # Data cleaning & validation pipeline
│   ├── feature_engineering.py
│   ├── model.py              # Model training, evaluation & benchmarking
│   ├── scorecard.py          # Score grid calibration
│   └── monitoring.py         # Production monitoring & alerting system
│
├── reports/
│   └── backtest_report.md    # Stability analysis across cohorts
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_model.py
│   └── test_monitoring.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/ntaccul_credit_scoring.git
cd ntaccul_credit_scoring
pip install -r requirements.txt
```

---

## Usage

### 1. Run the full pipeline

```python
from src.data_processing import DataPipeline
from src.feature_engineering import FeatureEngineer
from src.model import CreditScorer

# Load and clean data
pipeline = DataPipeline("data/raw/loan_data.csv")
df_clean = pipeline.run()

# Feature engineering
fe = FeatureEngineer(df_clean)
df_features = fe.transform()

# Train and evaluate model
scorer = CreditScorer(df_features)
scorer.train()
scorer.evaluate()
```

### 2. Run monitoring

```python
from src.monitoring import ProductionMonitor

monitor = ProductionMonitor(model_path="models/credit_scorer.pkl")
monitor.run(new_data="data/processed/batch_2025.csv")
# → Triggers alerts if drift detected
```

---

## Methodology

### Data
- **255,000 loan records** (NTACCUL portfolio, amounts in FCFA)
- Features: demographic, financial, and behavioural variables
- Target: binary default indicator (0 = performing, 1 = default)

### Feature Engineering
- Ratio-based features (debt-to-income, loan-to-value)
- Discretisation of continuous variables into risk bins
- Missing value imputation and outlier handling
- Variance-based feature selection

### Modelling
- **Primary model**: Logistic Regression (chosen for interpretability in regulatory context)
- **Benchmarked against**: Decision Tree, Random Forest, Gradient Boosting (XGBoost)
- **Evaluation**: ROC-AUC, Precision, Recall, F1-score, KS statistic
- **Calibration**: Score recalibration with percentage-based coefficient adjustment

### Scorecard
- Score grid mapping predicted probabilities to risk grades (1–10)
- Progressive default rate per risk class validated
- Production-ready scorecard with documented cutoffs

### Monitoring
- Automated drift detection on score distribution (PSI-based)
- Alert system triggered when AUC deviation exceeds threshold
- Cohort-based backtest: AUC maintained in [72.6%–73.5%] across 2020–2024 generations

---

## Tech Stack

| Tool | Usage |
|------|-------|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data manipulation |
| scikit-learn | Modelling & evaluation |
| XGBoost | Benchmarking |
| Matplotlib / Seaborn | Visualisation |
| pytest | Unit testing |
| Jupyter | Exploratory analysis |

---

## Backtest Summary

Stability analysis conducted across 5 application cohorts (2020–2024):

| Cohort | AUC | Score Distribution (grades 1–4) | Score Distribution (grades 8–10) |
|--------|-----|--------------------------------|----------------------------------|
| Reference | 73.9% | — | — |
| 2020 | 72.8% | — | — |
| 2021 | 73.1% | — | — |
| 2022 | 73.4% | — | — |
| 2023 | 73.5% | 14.8% | 52.3% |
| 2024 | 72.6% | 14.2% | 52.6% |

→ Model remains **stable and production-ready** across all cohorts.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Developed as part of a freelance data analytics engagement for NTACCUL, Cameroon.*
