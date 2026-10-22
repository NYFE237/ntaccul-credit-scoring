"""
test_serving.py
---------------
Checks the inference layer against the training pipeline, on synthetic data
that follows the real schema (the real loan_data.csv is not versioned).

Key guarantee: scoring a loan through the API gives the same features and the
same probability as the offline training pipeline (no train/serving skew).
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from fastapi.testclient import TestClient

import api.main as api_main
from src.data_processing import DataPipeline
from src.export_artifacts import export_artifacts
from src.feature_engineering import FeatureEngineer
from src.serving import InferencePreprocessor


# ── Synthetic data ─────────────────────────────────────────────────────────────

def make_loans(n: int = 6000, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    status = rng.choice(
        ["on_time", "1_month_late", "2_months_late", "3_months_late", "unknown"],
        n, p=[0.55, 0.2, 0.1, 0.1, 0.05],
    )
    income = rng.uniform(100_000, 2_000_000, n)
    amount = rng.uniform(50_000, 5_000_000, n)
    bill = rng.uniform(0, 600_000, n)
    pay = bill * rng.uniform(0, 1.2, n) * (rng.random(n) > 0.15)
    risk = {"on_time": 0, "1_month_late": 1, "2_months_late": 2,
            "3_months_late": 3, "unknown": 2}
    logit = (-3.2 + 0.6 * np.vectorize(risk.get)(status)
             + 1.2 * (amount / income > 4) + rng.normal(0, 0.8, n))
    df = pd.DataFrame({
        "loan_id": np.arange(n),
        "amount_fcfa": amount,
        "duration_months": rng.choice([6, 12, 24, 36], n),
        "age": rng.integers(18, 70, n).astype(float),
        "sex": rng.choice(["male", "female"], n),
        "marriage": rng.choice(["single", "married", "other"], n),
        "education": rng.choice(["primary", "secondary", "university"], n),
        "income_fcfa": income,
        "pay_status": status,
        "bill_amt": bill,
        "pay_amt": pay,
        "default": (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int),
    })
    # missing values and outliers, to exercise imputation and clipping
    for col in ["income_fcfa", "age", "bill_amt"]:
        df.loc[rng.choice(n, 120, replace=False), col] = np.nan
    df.loc[rng.choice(n, 40, replace=False), "amount_fcfa"] *= 20
    df.loc[rng.choice(n, 60, replace=False), "sex"] = np.nan
    return df


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    d = tmp_path_factory.mktemp("ntaccul")
    csv = d / "loan_data.csv"
    make_loans().to_csv(csv, index=False)
    export_artifacts(str(csv), str(d / "models"))
    return d


@pytest.fixture(scope="module")
def client(workdir):
    api_main.MODEL_DIR = workdir / "models"
    with TestClient(api_main.app) as c:
        yield c


GOOD = {
    "amount_fcfa": 1_500_000, "duration_months": 12, "age": 34,
    "sex": "female", "marriage": "married", "education": "secondary",
    "income_fcfa": 450_000, "pay_status": "on_time",
    "bill_amt": 120_000, "pay_amt": 100_000, "loan_id": "TEST-1",
}


# ── Parity with the training pipeline ──────────────────────────────────────────

def test_features_identical_to_training(workdir):
    csv = str(workdir / "loan_data.csv")
    raw = pd.read_csv(csv)
    train_feats = FeatureEngineer(DataPipeline(csv).run()).transform()
    train_feats = train_feats.drop(columns="default")

    pre = InferencePreprocessor.load(workdir / "models" / "preprocessing.json")
    served = pre.transform(raw.loc[train_feats.index].drop(columns=["loan_id", "default"]))

    assert list(served.columns) == list(train_feats.columns)
    np.testing.assert_allclose(
        served.to_numpy(dtype=float), train_feats.to_numpy(dtype=float), rtol=1e-9, atol=1e-9
    )


def test_single_row_equals_same_row_in_batch(workdir):
    pre = InferencePreprocessor.load(workdir / "models" / "preprocessing.json")
    raw = pd.read_csv(workdir / "loan_data.csv").drop(columns=["loan_id", "default"])
    batch = pre.transform(raw.iloc[:50])
    single = pre.transform(raw.iloc[[7]])
    np.testing.assert_allclose(single.to_numpy(float), batch.iloc[[7]].to_numpy(float))


# ── API ────────────────────────────────────────────────────────────────────────

def test_health(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_predict_returns_valid_score(client):
    r = client.post("/predict", json=GOOD)
    assert r.status_code == 200
    body = r.json()
    assert 0.0 <= body["probability_of_default"] <= 1.0
    assert 1 <= body["risk_grade"] <= 10
    assert body["risk_label"] in {"Low Risk", "Medium Risk", "High Risk"}
    assert body["loan_id"] == "TEST-1"


def test_api_probability_matches_offline_model(client, workdir):
    import pickle
    csv = str(workdir / "loan_data.csv")
    clean = DataPipeline(csv).run()
    feats = FeatureEngineer(clean).transform().drop(columns="default")
    model = pickle.load(open(workdir / "models" / "credit_scorer.pkl", "rb"))
    offline = model.predict_proba(feats)[:, 1]

    raw = pd.read_csv(csv).loc[feats.index].iloc[:25]
    payload = {"applications": [
        {k: (None if pd.isna(v) else v) for k, v in row.items() if k not in ("loan_id", "default")}
        for row in raw.to_dict("records")
    ]}
    # fields that are NaN in the raw file are required by the schema -> fill with defaults
    for a in payload["applications"]:
        a["income_fcfa"] = a["income_fcfa"] if a["income_fcfa"] is not None else 0
    rows = [a for a in payload["applications"]
            if None not in (a["age"], a["sex"], a["bill_amt"]) and a["income_fcfa"] != 0]
    idx = [i for i, a in enumerate(payload["applications"]) if a in rows]
    assert len(rows) >= 10, "test would be vacuous"

    r = client.post("/predict/batch", json={"applications": rows})
    assert r.status_code == 200, r.text
    got = [p["probability_of_default"] for p in r.json()["predictions"]]
    np.testing.assert_allclose(got, offline[idx], atol=1e-4)


def test_batch(client):
    r = client.post("/predict/batch", json={"applications": [GOOD, GOOD]})
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 2


def test_unknown_category_rejected(client):
    r = client.post("/predict", json={**GOOD, "education": "phd"})
    assert r.status_code == 422
    assert "Allowed" in r.json()["detail"]


def test_invalid_age_rejected(client):
    assert client.post("/predict", json={**GOOD, "age": 12}).status_code == 422


def test_riskier_profile_scores_higher(client):
    safe = client.post("/predict", json={**GOOD, "pay_status": "on_time"}).json()
    risky = client.post("/predict", json={
        **GOOD, "pay_status": "3_months_late", "pay_amt": 0, "amount_fcfa": 4_500_000,
    }).json()
    assert risky["probability_of_default"] > safe["probability_of_default"]


def test_metadata_lists_allowed_categories(client):
    body = client.get("/metadata").json()
    assert set(body["allowed_categories"]) == {"sex", "marriage", "education"}
    assert "model_name" in body
