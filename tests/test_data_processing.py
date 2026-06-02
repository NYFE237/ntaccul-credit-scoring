"""
test_data_processing.py
-----------------------
Unit tests for the DataPipeline class.
"""

import pytest
import pandas as pd
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.data_processing import DataPipeline, split_train_test


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_sample_df(n: int = 500, with_missing: bool = False, with_dupes: bool = False) -> pd.DataFrame:
    """Generate a minimal valid loan DataFrame for testing."""
    np.random.seed(42)
    df = pd.DataFrame({
        "loan_id": range(n),
        "amount_fcfa": np.random.uniform(50_000, 5_000_000, n),
        "duration_months": np.random.choice([6, 12, 24, 36], n),
        "age": np.random.randint(18, 65, n),
        "sex": np.random.choice(["male", "female"], n),
        "marriage": np.random.choice(["single", "married", "other"], n),
        "education": np.random.choice(["primary", "secondary", "university"], n),
        "income_fcfa": np.random.uniform(100_000, 2_000_000, n),
        "pay_status": np.random.choice(["on_time", "1_month_late", "2_months_late"], n),
        "bill_amt": np.random.uniform(0, 500_000, n),
        "pay_amt": np.random.uniform(0, 300_000, n),
        "default": np.random.choice([0, 1], n, p=[0.8, 0.2]),
    })
    if with_missing:
        df.loc[np.random.choice(n, 30, replace=False), "age"] = np.nan
        df.loc[np.random.choice(n, 20, replace=False), "income_fcfa"] = np.nan
    if with_dupes:
        df = pd.concat([df, df.iloc[:10]], ignore_index=True)
        df.loc[n:, "loan_id"] = df.loc[:9, "loan_id"].values
    return df


def save_temp_csv(df: pd.DataFrame, tmp_path) -> str:
    path = tmp_path / "test_loans.csv"
    df.to_csv(path, index=False)
    return str(path)


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestDataPipeline:

    def test_pipeline_runs_cleanly(self, tmp_path):
        df = make_sample_df()
        path = save_temp_csv(df, tmp_path)
        pipeline = DataPipeline(path)
        result = pipeline.run()
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0

    def test_removes_duplicates(self, tmp_path):
        df = make_sample_df(n=200, with_dupes=True)
        path = save_temp_csv(df, tmp_path)
        pipeline = DataPipeline(path)
        result = pipeline.run()
        assert pipeline.get_report()["duplicates_removed"] == 10
        assert result["loan_id"].nunique() == len(result)

    def test_handles_missing_values(self, tmp_path):
        df = make_sample_df(n=300, with_missing=True)
        path = save_temp_csv(df, tmp_path)
        pipeline = DataPipeline(path)
        result = pipeline.run()
        assert result.isnull().sum().sum() == 0

    def test_target_valid(self, tmp_path):
        df = make_sample_df()
        path = save_temp_csv(df, tmp_path)
        pipeline = DataPipeline(path)
        result = pipeline.run()
        assert set(result["default"].unique()).issubset({0, 1})

    def test_report_contains_keys(self, tmp_path):
        df = make_sample_df()
        path = save_temp_csv(df, tmp_path)
        pipeline = DataPipeline(path)
        pipeline.run()
        report = pipeline.get_report()
        for key in ["initial_shape", "duplicates_removed", "default_rate", "final_shape"]:
            assert key in report

    def test_raises_on_missing_columns(self, tmp_path):
        df = make_sample_df().drop(columns=["loan_id"])
        path = save_temp_csv(df, tmp_path)
        pipeline = DataPipeline(path)
        with pytest.raises(ValueError, match="Missing required columns"):
            pipeline.run()


class TestSplitTrainTest:

    def test_split_proportions(self, tmp_path):
        df = make_sample_df(n=1000)
        path = save_temp_csv(df, tmp_path)
        pipeline = DataPipeline(path)
        clean = pipeline.run()
        X_train, X_test, y_train, y_test = split_train_test(clean, test_size=0.2)
        assert abs(len(y_test) / (len(y_train) + len(y_test)) - 0.2) < 0.02

    def test_stratification_preserved(self, tmp_path):
        df = make_sample_df(n=1000)
        path = save_temp_csv(df, tmp_path)
        pipeline = DataPipeline(path)
        clean = pipeline.run()
        _, _, y_train, y_test = split_train_test(clean)
        train_rate = y_train.mean()
        test_rate = y_test.mean()
        assert abs(train_rate - test_rate) < 0.05
