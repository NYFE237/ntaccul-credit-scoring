"""
data_processing.py
------------------
Data ingestion, cleaning, validation, and quality control pipeline
for the NTACCUL credit scoring project.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)


# ── Constants ─────────────────────────────────────────────────────────────────

REQUIRED_COLUMNS = [
    "loan_id", "amount_fcfa", "duration_months", "age",
    "sex", "marriage", "education", "income_fcfa",
    "pay_status", "bill_amt", "pay_amt", "default"
]

NUMERIC_COLUMNS = [
    "amount_fcfa", "duration_months", "age",
    "income_fcfa", "bill_amt", "pay_amt"
]

CATEGORICAL_COLUMNS = ["sex", "marriage", "education", "pay_status"]

TARGET_COLUMN = "default"


# ── DataPipeline ───────────────────────────────────────────────────────────────

class DataPipeline:
    """
    End-to-end data cleaning and validation pipeline.

    Parameters
    ----------
    filepath : str
        Path to the raw CSV loan file.
    target_col : str
        Name of the binary target column (default: 'default').

    Examples
    --------
    >>> pipeline = DataPipeline("data/raw/loan_data.csv")
    >>> df_clean = pipeline.run()
    """

    def __init__(self, filepath: str, target_col: str = TARGET_COLUMN):
        self.filepath = filepath
        self.target_col = target_col
        self.df: Optional[pd.DataFrame] = None
        self._report: dict = {}

    # ── Public API ──────────────────────────────────────────────────────────

    def run(self) -> pd.DataFrame:
        """Execute the full cleaning pipeline and return a clean DataFrame."""
        logger.info("Starting data pipeline...")
        self.df = self._load()
        self._validate_schema()
        self._remove_duplicates()
        self._handle_missing_values()
        self._enforce_types()
        self._clip_outliers()
        self._validate_target()
        self._log_report()
        logger.info("Pipeline complete. Shape: %s", self.df.shape)
        return self.df.copy()

    def get_report(self) -> dict:
        """Return a summary report of all cleaning steps."""
        return self._report

    # ── Private methods ─────────────────────────────────────────────────────

    def _load(self) -> pd.DataFrame:
        logger.info("Loading data from %s", self.filepath)
        df = pd.read_csv(self.filepath, low_memory=False)
        self._report["initial_shape"] = df.shape
        logger.info("Loaded %d rows × %d columns", *df.shape)
        return df

    def _validate_schema(self) -> None:
        missing_cols = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        logger.info("Schema validation passed.")

    def _remove_duplicates(self) -> None:
        n_before = len(self.df)
        self.df = self.df.drop_duplicates(subset="loan_id")
        n_dropped = n_before - len(self.df)
        self._report["duplicates_removed"] = n_dropped
        if n_dropped > 0:
            logger.warning("Removed %d duplicate rows.", n_dropped)

    def _handle_missing_values(self) -> None:
        missing = self.df.isnull().sum()
        missing = missing[missing > 0]
        self._report["missing_before"] = missing.to_dict()

        # Numeric: impute with median
        for col in NUMERIC_COLUMNS:
            if col in self.df.columns and self.df[col].isnull().any():
                median_val = self.df[col].median()
                self.df[col] = self.df[col].fillna(median_val)
                logger.info("Imputed '%s' with median=%.2f", col, median_val)

        # Categorical: impute with mode
        for col in CATEGORICAL_COLUMNS:
            if col in self.df.columns and self.df[col].isnull().any():
                mode_val = self.df[col].mode()[0]
                self.df[col] = self.df[col].fillna(mode_val)
                logger.info("Imputed '%s' with mode=%s", col, mode_val)

        self._report["missing_after"] = self.df.isnull().sum()[self.df.isnull().sum() > 0].to_dict()

    def _enforce_types(self) -> None:
        for col in NUMERIC_COLUMNS:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
        for col in CATEGORICAL_COLUMNS:
            if col in self.df.columns:
                self.df[col] = self.df[col].astype(str).str.strip().str.lower()
        logger.info("Data types enforced.")

    def _clip_outliers(self, lower_q: float = 0.01, upper_q: float = 0.99) -> None:
        clipped = {}
        for col in NUMERIC_COLUMNS:
            if col not in self.df.columns:
                continue
            lo = self.df[col].quantile(lower_q)
            hi = self.df[col].quantile(upper_q)
            n_clipped = ((self.df[col] < lo) | (self.df[col] > hi)).sum()
            self.df[col] = self.df[col].clip(lo, hi)
            if n_clipped > 0:
                clipped[col] = int(n_clipped)
        self._report["outliers_clipped"] = clipped
        logger.info("Outlier clipping done: %s", clipped)

    def _validate_target(self) -> None:
        if self.target_col not in self.df.columns:
            raise ValueError(f"Target column '{self.target_col}' not found.")
        invalid = ~self.df[self.target_col].isin([0, 1])
        n_invalid = invalid.sum()
        if n_invalid > 0:
            logger.warning("Dropping %d rows with invalid target values.", n_invalid)
            self.df = self.df[~invalid]
        default_rate = self.df[self.target_col].mean()
        self._report["default_rate"] = round(default_rate, 4)
        self._report["final_shape"] = self.df.shape
        logger.info("Default rate: %.2f%%", default_rate * 100)

    def _log_report(self) -> None:
        logger.info("=== Pipeline Report ===")
        for key, val in self._report.items():
            logger.info("  %s: %s", key, val)


# ── Utility functions ──────────────────────────────────────────────────────────

def split_train_test(
    df: pd.DataFrame,
    target_col: str = TARGET_COLUMN,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Stratified train/test split preserving default rate balance.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target_col, "loan_id"], errors="ignore")
    y = df[target_col]

    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
