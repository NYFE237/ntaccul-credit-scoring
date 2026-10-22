"""
serving.py
----------
Inference-time preprocessing for the NTACCUL credit scoring model.

Why this module exists
----------------------
``DataPipeline`` and ``FeatureEngineer`` compute statistics on the batch they
receive (outlier quantiles, medians, quintile bins, dummy columns). That is
correct for training, but wrong for scoring a single loan application:

* ``pd.qcut`` on one row cannot produce quintiles,
* ``pd.get_dummies`` on one row produces a different set of columns,
* clipping / imputation would use the statistics of the request itself.

``InferencePreprocessor`` replays exactly the same transformations using
statistics **frozen at training time** (see ``src/export_artifacts.py``).
A parity test (tests/test_serving.py) checks that the output is identical to
``FeatureEngineer.transform()`` on the training data.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

NUMERIC_COLUMNS = [
    "amount_fcfa", "duration_months", "age",
    "income_fcfa", "bill_amt", "pay_amt",
]
CATEGORICAL_COLUMNS = ["sex", "marriage", "education", "pay_status"]
DUMMY_COLUMNS = ["sex", "marriage", "education"]  # one-hot encoded in FeatureEngineer

PAY_STATUS_MAP = {
    "on_time": 0, "1_month_late": 1, "2_months_late": 2,
    "3_months_late": 3, "4_months_late": 4, "5_months_late": 5,
    "unknown": 2,
}
EPS = 1e-6


class InferencePreprocessor:
    """
    Turn raw loan applications into the exact feature matrix the model was
    trained on, using statistics frozen at training time.

    ``state`` keys
    --------------
    bounds          {col: [low, high]}  1st / 99th percentile clip bounds
    medians         {col: value}        numeric imputation values
    modes           {col: value}        categorical imputation values
    amount_edges    [6 floats]          quintile edges of ``amount_fcfa``
    categories      {col: [values]}     categories seen in training
    feature_columns [str]               final model input columns, in order
    """

    def __init__(self, state: Dict):
        self.state = state
        self.feature_columns: List[str] = list(state["feature_columns"])

    # ── Persistence ──────────────────────────────────────────────────────────

    @classmethod
    def load(cls, path: Union[str, Path]) -> "InferencePreprocessor":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def save(self, path: Union[str, Path]) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.state, f, indent=2)

    # ── Public API ───────────────────────────────────────────────────────────

    def transform(self, records: Union[pd.DataFrame, Sequence[Dict]]) -> pd.DataFrame:
        """
        Build model features from raw applications.

        Raises
        ------
        ValueError
            If a categorical value was never seen in training, or if a feature
            cannot be computed (e.g. age outside the supported bins).
        """
        df = records.copy() if isinstance(records, pd.DataFrame) else pd.DataFrame(list(records))
        df = df.reset_index(drop=True)

        df = self._clean(df)
        df = self._engineer(df)
        return self._align(df)

    # ── Step 1: same cleaning as DataPipeline, with frozen statistics ────────

    def _clean(self, df: pd.DataFrame) -> pd.DataFrame:
        s = self.state

        for col in NUMERIC_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
            lo, hi = s["bounds"][col]
            df[col] = (
                pd.to_numeric(df[col], errors="coerce")
                .fillna(s["medians"][col])
                .clip(lo, hi)
            )

        for col in CATEGORICAL_COLUMNS:
            if col not in df.columns:
                df[col] = np.nan
            df[col] = df[col].fillna(s["modes"][col]).astype(str).str.strip().str.lower()

        for col in DUMMY_COLUMNS:
            unseen = sorted(set(df[col]) - set(s["categories"][col]))
            if unseen:
                raise ValueError(
                    f"Unknown value(s) {unseen} for '{col}'. "
                    f"Allowed: {s['categories'][col]}"
                )
        return df

    # ── Step 2: same features as FeatureEngineer ─────────────────────────────

    def _engineer(self, df: pd.DataFrame) -> pd.DataFrame:
        out = pd.DataFrame(index=df.index)

        # Ratios
        out["income_fcfa"] = df["income_fcfa"]  # kept raw by FeatureEngineer
        out["debt_to_income"] = df["bill_amt"] / (df["income_fcfa"] + EPS)
        out["loan_to_income"] = df["amount_fcfa"] / (df["income_fcfa"] + EPS)
        out["repayment_ratio"] = df["pay_amt"] / (df["bill_amt"] + EPS)

        # Discretisation (bins identical to FeatureEngineer)
        out["age_bin"] = pd.cut(
            df["age"], bins=[0, 25, 35, 45, 55, 100],
            labels=[1, 2, 3, 4, 5], right=False,
        ).astype(float)

        edges = np.array(self.state["amount_edges"], dtype=float)
        edges[0], edges[-1] = -np.inf, np.inf
        out["amount_bin"] = pd.cut(
            df["amount_fcfa"], bins=edges, labels=[1, 2, 3, 4, 5],
            right=True, include_lowest=True,
        ).astype(float)

        out["duration_bin"] = pd.cut(
            df["duration_months"], bins=[0, 6, 12, 24, 36, 9999],
            labels=[1, 2, 3, 4, 5], right=False,
        ).astype(float)

        # Behavioural
        out["pay_risk_score"] = df["pay_status"].map(PAY_STATUS_MAP).fillna(2)
        out["zero_payment_flag"] = (
            (df["pay_amt"] == 0) & (df["bill_amt"] > 0)
        ).astype(int)

        # One-hot (drop_first) -> reproduce the training columns by name
        for col in self.feature_columns:
            for base in DUMMY_COLUMNS:
                prefix = f"{base}_"
                if col.startswith(prefix):
                    out[col] = (df[base] == col[len(prefix):]).astype(bool)
        return out

    # ── Step 3: strict alignment with the model input ────────────────────────

    def _align(self, df: pd.DataFrame) -> pd.DataFrame:
        missing = [c for c in self.feature_columns if c not in df.columns]
        if missing:
            raise ValueError(f"Features could not be built: {missing}")
        X = df[self.feature_columns]
        bad = X.columns[X.isna().any()].tolist()
        if bad:
            raise ValueError(
                f"Invalid input: could not compute {bad} "
                "(value outside the supported range, e.g. age < 0 or >= 100)."
            )
        return X
