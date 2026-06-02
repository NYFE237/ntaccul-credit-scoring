"""
feature_engineering.py
-----------------------
Feature engineering pipeline for the NTACCUL credit scoring model.
Covers ratio creation, discretisation, encoding, and feature selection.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


# ── FeatureEngineer ────────────────────────────────────────────────────────────

class FeatureEngineer:
    """
    Transforms clean loan data into model-ready features.

    Steps
    -----
    1. Ratio-based features (financial ratios)
    2. Discretisation of continuous variables into risk bins
    3. Lag / behavioural features from payment history
    4. One-hot encoding of categorical variables
    5. Variance-based feature selection

    Parameters
    ----------
    df : pd.DataFrame
        Clean DataFrame from DataPipeline.
    target_col : str
        Target column name (excluded from transformations).
    """

    def __init__(self, df: pd.DataFrame, target_col: str = "default"):
        self.df = df.copy()
        self.target_col = target_col
        self.feature_names_: Optional[List[str]] = None

    def transform(self) -> pd.DataFrame:
        """Run all feature engineering steps and return transformed DataFrame."""
        logger.info("Starting feature engineering...")
        self._create_ratio_features()
        self._discretise_variables()
        self._create_behavioural_features()
        self._encode_categoricals()
        self._drop_raw_columns()
        self.feature_names_ = [
            c for c in self.df.columns if c != self.target_col
        ]
        logger.info("Feature engineering complete. %d features created.", len(self.feature_names_))
        return self.df.copy()

    # ── Step 1: Financial ratios ─────────────────────────────────────────────

    def _create_ratio_features(self) -> None:
        """Create debt-to-income and loan-utilisation ratios."""
        eps = 1e-6  # avoid division by zero

        if "bill_amt" in self.df.columns and "income_fcfa" in self.df.columns:
            self.df["debt_to_income"] = self.df["bill_amt"] / (self.df["income_fcfa"] + eps)

        if "amount_fcfa" in self.df.columns and "income_fcfa" in self.df.columns:
            self.df["loan_to_income"] = self.df["amount_fcfa"] / (self.df["income_fcfa"] + eps)

        if "pay_amt" in self.df.columns and "bill_amt" in self.df.columns:
            self.df["repayment_ratio"] = self.df["pay_amt"] / (self.df["bill_amt"] + eps)

        logger.info("Ratio features created.")

    # ── Step 2: Discretisation ───────────────────────────────────────────────

    def _discretise_variables(self) -> None:
        """Bin continuous variables into ordinal risk categories."""

        # Age → risk bins
        if "age" in self.df.columns:
            self.df["age_bin"] = pd.cut(
                self.df["age"],
                bins=[0, 25, 35, 45, 55, 100],
                labels=[1, 2, 3, 4, 5],
                right=False
            ).astype(float)

        # Loan amount → size category
        if "amount_fcfa" in self.df.columns:
            self.df["amount_bin"] = pd.qcut(
                self.df["amount_fcfa"],
                q=5,
                labels=[1, 2, 3, 4, 5],
                duplicates="drop"
            ).astype(float)

        # Loan duration → term category
        if "duration_months" in self.df.columns:
            self.df["duration_bin"] = pd.cut(
                self.df["duration_months"],
                bins=[0, 6, 12, 24, 36, 9999],
                labels=[1, 2, 3, 4, 5],
                right=False
            ).astype(float)

        logger.info("Discretisation complete.")

    # ── Step 3: Behavioural features ─────────────────────────────────────────

    def _create_behavioural_features(self) -> None:
        """Derive features from payment status and history."""

        if "pay_status" in self.df.columns:
            # Map payment status to risk score
            status_map = {
                "on_time": 0, "1_month_late": 1, "2_months_late": 2,
                "3_months_late": 3, "4_months_late": 4, "5_months_late": 5,
                "unknown": 2
            }
            self.df["pay_risk_score"] = (
                self.df["pay_status"].map(status_map).fillna(2)
            )

        if "pay_amt" in self.df.columns and "bill_amt" in self.df.columns:
            # Flag zero payment on non-zero bill
            self.df["zero_payment_flag"] = (
                (self.df["pay_amt"] == 0) & (self.df["bill_amt"] > 0)
            ).astype(int)

        logger.info("Behavioural features created.")

    # ── Step 4: Encoding ──────────────────────────────────────────────────────

    def _encode_categoricals(self) -> None:
        """One-hot encode remaining categorical columns."""
        cat_cols = [
            c for c in ["sex", "marriage", "education"]
            if c in self.df.columns
        ]
        if cat_cols:
            self.df = pd.get_dummies(self.df, columns=cat_cols, drop_first=True)
        logger.info("Categorical encoding done. Columns: %s", cat_cols)

    # ── Step 5: Drop raw columns ─────────────────────────────────────────────

    def _drop_raw_columns(self) -> None:
        """Remove original columns replaced by engineered features."""
        cols_to_drop = [
            "loan_id", "age", "duration_months", "amount_fcfa",
            "pay_status", "bill_amt", "pay_amt"
        ]
        self.df = self.df.drop(
            columns=[c for c in cols_to_drop if c in self.df.columns],
            errors="ignore"
        )

    def get_feature_names(self) -> List[str]:
        """Return the list of final feature names after transformation."""
        if self.feature_names_ is None:
            raise RuntimeError("Call transform() first.")
        return self.feature_names_
