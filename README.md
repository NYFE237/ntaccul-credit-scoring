# Credit Scoring Model — NTACCUL Microfinance (Cameroon)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-orange?logo=scikit-learn)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688?logo=fastapi)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## Overview

End-to-end **credit scoring model** developed for **NTACCUL**, a microfinance institution in Cameroon.
The goal is to predict the probability of loan default on a portfolio of **255,000 loan files** (amounts in FCFA), enabling proactive risk management and optimised lending policy decisions.

The project covers the full modelling lifecycle, up to a deployable scoring API:

```
Data Ingestion → Feature Engineering → Model Training & Benchmarking
→ Scorecard (risk grades 1–10) → Scoring API → Monitoring (PSI, AUC)
```

---

## Key Results

| Metric | Baseline Model | Final Model |
|--------|---------------|-------------|
| Default Detection Rate | 1× (reference) | **+910%** |
| At-Risk Cases Identified | — | **3,030 / 51,000** |
| AUC (reference cohort) | — | **73.9%** |
| AUC stability (2020–2024) | — | [72.6% – 73.5%] |
| Monitoring | None | PSI + AUC alerts |

---

## Project Structure

```
ntaccul-credit-scoring/
│
├── data/
│   ├── raw/                   # Original loan files (not versioned)
│   └── processed/             # Cleaned & feature-engineered datasets (not versioned)
│
├── src/
│   ├── data_processing.py     # Cleaning & validation pipeline
│   ├── feature_engineering.py # Ratios, risk bins, behavioural features, encoding
│   ├── model.py               # Training, benchmarking, evaluation, persistence
│   ├── scorecard.py           # Probability → risk grade (1–10) mapping
│   ├── monitoring.py          # PSI / AUC drift monitoring and alerts
│   ├── serving.py             # Inference-time preprocessing (frozen training statistics)
│   └── export_artifacts.py    # Train + export model, preprocessing state, metadata
│
├── api/
│   └── main.py                # FastAPI scoring service
│
├── reports/
│   └── backtest_report.md     # Stability analysis across cohorts
│
├── tests/
│   ├── test_data_processing.py
│   ├── test_monitoring.py
│   └── test_serving.py        # Parity between API and training pipeline + API tests
│
├── Dockerfile
├── requirements.txt           # Training / analysis
├── requirements-api.txt       # Serving
└── README.md
```

---

## Installation

```bash
git clone https://github.com/NYFE237/ntaccul-credit-scoring.git
cd ntaccul-credit-scoring
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
scorer.save("models/credit_scorer.pkl")
```

### 2. Monitor a new batch

```python
import json
import numpy as np
from src.monitoring import ProductionMonitor

meta = json.load(open("models/metadata.json"))   # written by export_artifacts
monitor = ProductionMonitor(
    reference_probs=np.load("models/reference_probs.npy"),
    reference_auc=meta["test_metrics"]["auc"],
)
report = monitor.run(new_probs=new_probs, y_true=y_true, batch_label="2025-Q1")
# → alerts if PSI ≥ 0.10 / 0.25 or if AUC drops by more than 2 points
```

---

## Scoring API

A FastAPI service returns the probability of default and a 1–10 risk grade for a loan application.

### 1. Train and export the artifacts

The raw data and the trained model are not versioned. Generate the artifacts locally:

```bash
python -m src.export_artifacts --data data/raw/loan_data.csv --out models
```

This writes `credit_scorer.pkl`, `preprocessing.json` (statistics frozen at training time),
`reference_probs.npy` (reference distribution for PSI) and `metadata.json`.

> `DataPipeline` and `FeatureEngineer` compute statistics on the batch they receive (quantiles, medians,
> dummy columns), so they cannot score a single loan. `src/serving.py` replays the same transformations
> with the frozen statistics, and `tests/test_serving.py` checks that features and probabilities are
> identical to the training pipeline.

### 2. Run the API

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload        # Swagger UI: http://localhost:8000/docs
```

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "amount_fcfa": 1500000, "duration_months": 12, "age": 34,
  "sex": "female", "marriage": "married", "education": "secondary",
  "income_fcfa": 450000, "pay_status": "on_time",
  "bill_amt": 120000, "pay_amt": 100000
}'
# {"loan_id":null,"probability_of_default":0.37,"risk_grade":4,"risk_label":"Low Risk"}
```

| Endpoint | Purpose |
|----------|---------|
| `GET /health` | Liveness check |
| `GET /metadata` | Model info and accepted categorical values |
| `POST /predict` | Score one application |
| `POST /predict/batch` | Score up to 1,000 applications |

Unknown categories and out-of-range values are rejected with a `422` and an explicit message.

### 3. Docker

```bash
docker build -t ntaccul-api .
docker run -p 7860:7860 ntaccul-api   # http://localhost:7860/docs
```

The image installs the scikit-learn / pandas / numpy versions recorded in `models/metadata.json`,
since a pickled model is only guaranteed to load under the version that produced it.

---

## Methodology

### Data
- **255,000 loan records** (NTACCUL portfolio, amounts in FCFA)
- Features: demographic, financial, and behavioural variables
- Target: binary default indicator (0 = performing, 1 = default)

### Data processing
- Schema validation and duplicate removal on `loan_id`
- Median (numeric) and mode (categorical) imputation
- Outlier clipping at the 1st / 99th percentiles
- Target validation (binary values only)

### Feature engineering
- Ratios: `debt_to_income`, `loan_to_income`, `repayment_ratio`
- Ordinal risk bins for age, loan amount (quintiles) and duration
- Behavioural features: `pay_risk_score` from payment status, `zero_payment_flag`
- One-hot encoding of sex, marital status and education

### Modelling
- **Primary model**: Logistic Regression (standardised inputs, balanced class weights), chosen for interpretability in a regulated context
- **Benchmarked against**: Decision Tree, Random Forest and Gradient Boosting (scikit-learn), 5-fold stratified cross-validation on ROC-AUC
- **Selection rule**: Logistic Regression is retained when its cross-validated AUC is within 0.02 of the best model; otherwise the best model is used
- **Evaluation**: ROC-AUC, precision / recall / F1 on the default class at the F2-optimal threshold (recall-oriented), confusion matrix

### Scorecard
- Probabilities mapped to risk grades 1–10 using deciles of the training score distribution
- Grades 1–4 low risk, 5–7 medium, 8–10 high
- Automatic check that default rates increase with the grade

### Monitoring
- Population Stability Index on the score distribution (warning ≥ 0.10, alert ≥ 0.25)
- AUC degradation alert when AUC drops by more than 2 points versus the reference
- Cohort-based backtest: AUC maintained in [72.6%–73.5%] across 2020–2024 generations

---

## Tech Stack

| Tool | Usage |
|------|-------|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data manipulation |
| scikit-learn | Modelling & evaluation |
| FastAPI / Uvicorn | Scoring API |
| Docker | Packaging |
| pytest | Unit and integration tests |

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

→ AUC remains within a 1-point band across all cohorts.

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

*Developed as part of a freelance data analytics engagement for NTACCUL, Cameroon.*
