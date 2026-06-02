"""
scorecard.py
------------
Score grid calibration, risk grade assignment, and scorecard generation
for the NTACCUL credit scoring project.

Maps predicted default probabilities to an interpretable 1–10 risk scale.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import logging
from typing import Optional

logger = logging.getLogger(__name__)


# ── Scorecard ──────────────────────────────────────────────────────────────────

class Scorecard:
    """
    Maps predicted default probabilities to a 1–10 risk grade scale.

    - Grade 1–4  : low risk (performing loans)
    - Grade 5–7  : medium risk (watch list)
    - Grade 8–10 : high risk (potential default)

    Calibration ensures monotonic progression of default rates
    across grades — a key regulatory requirement.

    Parameters
    ----------
    n_grades : int
        Number of risk grades (default: 10).
    """

    GRADE_LABELS = {
        (1, 4): "Low Risk",
        (5, 7): "Medium Risk",
        (8, 10): "High Risk",
    }

    def __init__(self, n_grades: int = 10):
        self.n_grades = n_grades
        self.thresholds_: Optional[np.ndarray] = None
        self.grade_stats_: Optional[pd.DataFrame] = None

    # ── Public API ──────────────────────────────────────────────────────────

    def fit(self, y_prob: np.ndarray, y_true: np.ndarray) -> "Scorecard":
        """
        Calibrate grade thresholds on training probabilities.

        Parameters
        ----------
        y_prob : np.ndarray
            Predicted default probabilities.
        y_true : np.ndarray
            True binary labels.
        """
        self.thresholds_ = np.percentile(
            y_prob,
            np.linspace(0, 100, self.n_grades + 1)
        )
        grades = self._assign_grades(y_prob)
        self.grade_stats_ = self._compute_grade_stats(grades, y_true, y_prob)
        self._validate_monotonicity()
        logger.info("Scorecard calibrated on %d observations.", len(y_prob))
        return self

    def transform(self, y_prob: np.ndarray) -> np.ndarray:
        """Assign risk grades to new predicted probabilities."""
        if self.thresholds_ is None:
            raise RuntimeError("Call fit() first.")
        return self._assign_grades(y_prob)

    def fit_transform(self, y_prob: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(y_prob, y_true)
        return self.transform(y_prob)

    def get_scorecard_table(self) -> pd.DataFrame:
        """Return the full scorecard table with default rates per grade."""
        if self.grade_stats_ is None:
            raise RuntimeError("Call fit() first.")
        return self.grade_stats_.copy()

    def get_risk_label(self, grade: int) -> str:
        """Return the risk label for a given grade."""
        for (lo, hi), label in self.GRADE_LABELS.items():
            if lo <= grade <= hi:
                return label
        return "Unknown"

    def summary(self) -> None:
        """Print a formatted scorecard summary."""
        if self.grade_stats_ is None:
            raise RuntimeError("Call fit() first.")
        print("\n" + "=" * 60)
        print("NTACCUL CREDIT SCORECARD — Risk Grade Summary")
        print("=" * 60)
        print(self.grade_stats_.to_string(index=False))
        print("=" * 60 + "\n")

    # ── Private methods ─────────────────────────────────────────────────────

    def _assign_grades(self, y_prob: np.ndarray) -> np.ndarray:
        """Map probabilities to integer grades 1–10."""
        grades = np.digitize(y_prob, self.thresholds_[1:-1]) + 1
        return np.clip(grades, 1, self.n_grades)

    def _compute_grade_stats(
        self,
        grades: np.ndarray,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> pd.DataFrame:
        """Compute population and default rate per grade."""
        df = pd.DataFrame({
            "grade": grades,
            "default": y_true,
            "prob": y_prob,
        })
        stats = (
            df.groupby("grade")
            .agg(
                count=("default", "count"),
                n_defaults=("default", "sum"),
                avg_prob=("prob", "mean"),
            )
            .reset_index()
        )
        stats["default_rate_%"] = (stats["n_defaults"] / stats["count"] * 100).round(2)
        stats["pct_population_%"] = (stats["count"] / stats["count"].sum() * 100).round(2)
        stats["risk_label"] = stats["grade"].apply(self.get_risk_label)
        return stats[[
            "grade", "risk_label", "count", "pct_population_%",
            "n_defaults", "default_rate_%", "avg_prob"
        ]]

    def _validate_monotonicity(self) -> None:
        """Warn if default rates are not monotonically increasing across grades."""
        rates = self.grade_stats_["default_rate_%"].values
        is_monotone = all(rates[i] <= rates[i + 1] for i in range(len(rates) - 1))
        if is_monotone:
            logger.info("Monotonicity check PASSED — default rates increase with grade.")
        else:
            logger.warning(
                "Monotonicity check FAILED — some grades have non-increasing default rates. "
                "Consider re-calibrating thresholds."
            )
