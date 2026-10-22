"""
NTACCUL credit scoring API.

Run locally:   uvicorn api.main:app --reload
Swagger UI:    http://localhost:8000/docs

Artifacts are read from $MODEL_DIR (default: ./models), produced by:
    python -m src.export_artifacts --data data/raw/loan_data.csv --out models
"""

import json
import logging
import os
import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import sklearn
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from src.scorecard import Scorecard
from src.serving import InferencePreprocessor

logger = logging.getLogger("ntaccul.api")
MODEL_DIR = Path(os.getenv("MODEL_DIR", "models"))
MAX_BATCH = 1000


# ── Schemas ────────────────────────────────────────────────────────────────────

class LoanApplication(BaseModel):
    """Raw loan application: the columns of loan_data.csv, without `default`."""

    amount_fcfa: float = Field(..., gt=0, examples=[1_500_000], description="Loan amount (FCFA)")
    duration_months: float = Field(..., gt=0, le=120, examples=[12])
    age: float = Field(..., ge=18, lt=100, examples=[34])
    sex: str = Field(..., examples=["female"])
    marriage: str = Field(..., examples=["married"])
    education: str = Field(..., examples=["secondary"])
    income_fcfa: float = Field(..., ge=0, examples=[450_000], description="Monthly income (FCFA)")
    pay_status: str = Field(
        "unknown",
        examples=["on_time"],
        description="on_time, 1_month_late ... 5_months_late, unknown",
    )
    bill_amt: float = Field(0, ge=0, examples=[120_000])
    pay_amt: float = Field(0, ge=0, examples=[100_000])
    loan_id: Optional[str] = Field(None, description="Echoed back, not used by the model")


class Prediction(BaseModel):
    loan_id: Optional[str] = None
    probability_of_default: float
    risk_grade: int = Field(..., description="1 (lowest risk) to 10 (highest risk)")
    risk_label: str


class BatchRequest(BaseModel):
    applications: List[LoanApplication] = Field(..., min_length=1, max_length=MAX_BATCH)


class BatchResponse(BaseModel):
    predictions: List[Prediction]


# ── App lifecycle ──────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    with open(MODEL_DIR / "credit_scorer.pkl", "rb") as f:
        app.state.model = pickle.load(f)
    app.state.pre = InferencePreprocessor.load(MODEL_DIR / "preprocessing.json")
    with open(MODEL_DIR / "metadata.json", encoding="utf-8") as f:
        app.state.meta = json.load(f)

    scorecard = Scorecard()
    scorecard.thresholds_ = np.array(app.state.pre.state["scorecard_thresholds"])
    app.state.scorecard = scorecard

    trained = app.state.meta.get("versions", {}).get("scikit-learn")
    if trained and trained != sklearn.__version__:
        logger.warning(
            "scikit-learn mismatch: model trained with %s, serving with %s. "
            "Pin the training version in requirements-api.txt.",
            trained, sklearn.__version__,
        )
    yield


app = FastAPI(
    title="NTACCUL Credit Scoring API",
    description="Probability of default and risk grade (1-10) for microfinance loan applications.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Helpers ────────────────────────────────────────────────────────────────────

def _score(request: Request, applications: List[LoanApplication]) -> List[Prediction]:
    state = request.app.state
    raw = pd.DataFrame([a.model_dump(exclude={"loan_id"}) for a in applications])
    try:
        X = state.pre.transform(raw)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))

    probs = state.model.predict_proba(X)[:, 1]
    grades = state.scorecard.transform(probs)
    return [
        Prediction(
            loan_id=a.loan_id,
            probability_of_default=round(float(p), 4),
            risk_grade=int(g),
            risk_label=state.scorecard.get_risk_label(int(g)),
        )
        for a, p, g in zip(applications, probs, grades)
    ]


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/metadata")
def metadata(request: Request):
    """Model info and the categorical values accepted by /predict."""
    state = request.app.state
    return {
        **state.meta,
        "allowed_categories": state.pre.state["categories"],
        "pay_status_values": [
            "on_time", "1_month_late", "2_months_late", "3_months_late",
            "4_months_late", "5_months_late", "unknown",
        ],
    }


@app.post("/predict", response_model=Prediction)
def predict(application: LoanApplication, request: Request):
    return _score(request, [application])[0]


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest, request: Request):
    return BatchResponse(predictions=_score(request, batch.applications))
