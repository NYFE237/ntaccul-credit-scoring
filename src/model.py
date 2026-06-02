"""
model.py
--------
Model training, benchmarking, evaluation, and persistence
for the NTACCUL credit scoring project.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import pickle
import logging
from typing import Optional, Dict

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report,
    confusion_matrix, precision_recall_curve
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

logger = logging.getLogger(__name__)


# ── CreditScorer ───────────────────────────────────────────────────────────────

class CreditScorer:
    """
    Credit default scoring model with benchmarking and evaluation.

    The primary model is Logistic Regression, chosen for:
    - Interpretability (required in regulated/financial contexts)
    - Probability calibration (direct PD estimation)
    - Auditable coefficients

    ML models (Decision Tree, Random Forest, Gradient Boosting) are
    benchmarked for comparison; Logistic Regression is retained if its
    AUC is within an acceptable gap vs the best ML model.

    Parameters
    ----------
    df : pd.DataFrame
        Feature-engineered DataFrame (output of FeatureEngineer).
    target_col : str
        Name of the binary target column.
    random_state : int
        Reproducibility seed.
    """

    BENCHMARK_MODELS = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(
                max_iter=1000, class_weight="balanced", random_state=42
            ))
        ]),
        "decision_tree": DecisionTreeClassifier(
            max_depth=6, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=8,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4,
            learning_rate=0.05, random_state=42
        ),
    }

    def __init__(
        self,
        df: pd.DataFrame,
        target_col: str = "default",
        random_state: int = 42,
    ):
        self.target_col = target_col
        self.random_state = random_state
        self.X = df.drop(columns=[target_col])
        self.y = df[target_col]
        self.model: Optional[object] = None
        self.benchmark_results_: Dict[str, float] = {}
        self.best_model_name_: Optional[str] = None
        self.X_train = self.X_test = self.y_train = self.y_test = None

    # ── Public API ──────────────────────────────────────────────────────────

    def train(self, test_size: float = 0.2) -> "CreditScorer":
        """Train primary model and benchmark against ML alternatives."""
        from sklearn.model_selection import train_test_split

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=test_size,
            stratify=self.y,
            random_state=self.random_state,
        )
        logger.info("Training set: %d | Test set: %d", len(self.y_train), len(self.y_test))
        self._benchmark()
        self._select_and_fit_final()
        return self

    def evaluate(self) -> Dict[str, float]:
        """Evaluate the final model on the test set."""
        if self.model is None:
            raise RuntimeError("Call train() first.")

        y_prob = self.model.predict_proba(self.X_test)[:, 1]
        y_pred = (y_prob >= self._find_optimal_threshold(y_prob)).astype(int)

        auc = roc_auc_score(self.y_test, y_prob)
        report = classification_report(self.y_test, y_pred, output_dict=True)

        results = {
            "auc": round(auc, 4),
            "precision_default": round(report["1"]["precision"], 4),
            "recall_default": round(report["1"]["recall"], 4),
            "f1_default": round(report["1"]["f1-score"], 4),
            "accuracy": round(report["accuracy"], 4),
        }

        logger.info("=== Final Model Evaluation ===")
        for k, v in results.items():
            logger.info("  %s: %s", k, v)

        self._log_confusion_matrix(y_pred)
        return results

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted default probabilities for new data."""
        if self.model is None:
            raise RuntimeError("Call train() first.")
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: str = "models/credit_scorer.pkl") -> None:
        """Persist the trained model to disk."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.model, f)
        logger.info("Model saved to %s", path)

    @classmethod
    def load(cls, path: str) -> object:
        """Load a persisted model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)

    # ── Private methods ─────────────────────────────────────────────────────

    def _benchmark(self, cv: int = 5) -> None:
        """Cross-validate all benchmark models and store AUC scores."""
        logger.info("Benchmarking %d models (CV=%d)...", len(self.BENCHMARK_MODELS), cv)
        cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        for name, model in self.BENCHMARK_MODELS.items():
            scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=cv_splitter, scoring="roc_auc", n_jobs=-1
            )
            mean_auc = scores.mean()
            self.benchmark_results_[name] = round(mean_auc, 4)
            logger.info("  %-30s AUC = %.4f (±%.4f)", name, mean_auc, scores.std())

    def _select_and_fit_final(self, explainability_gap: float = 0.02) -> None:
        """
        Select final model.

        Retain Logistic Regression if its AUC is within `explainability_gap`
        of the best ML model — preserving interpretability in a regulated context.
        """
        lr_auc = self.benchmark_results_.get("logistic_regression", 0)
        best_auc = max(self.benchmark_results_.values())
        best_name = max(self.benchmark_results_, key=self.benchmark_results_.get)

        if best_auc - lr_auc <= explainability_gap:
            self.best_model_name_ = "logistic_regression"
            logger.info(
                "Logistic Regression selected (AUC gap=%.4f ≤ %.4f threshold).",
                best_auc - lr_auc, explainability_gap
            )
        else:
            self.best_model_name_ = best_name
            logger.info(
                "Best ML model selected: %s (AUC=%.4f, gap=%.4f > threshold).",
                best_name, best_auc, best_auc - lr_auc
            )

        self.model = self.BENCHMARK_MODELS[self.best_model_name_]
        self.model.fit(self.X_train, self.y_train)
        logger.info("Final model fitted on full training set.")

    def _find_optimal_threshold(self, y_prob: np.ndarray, beta: float = 2.0) -> float:
        """
        Find the decision threshold maximising F-beta score.
        Beta > 1 prioritises recall (detecting defaults).
        """
        precisions, recalls, thresholds = precision_recall_curve(self.y_test, y_prob)
        f_beta = ((1 + beta**2) * precisions * recalls) / (beta**2 * precisions + recalls + 1e-9)
        best_idx = np.argmax(f_beta[:-1])
        threshold = thresholds[best_idx]
        logger.info("Optimal threshold: %.4f (F%.1f maximisation)", threshold, beta)
        return threshold

    def _log_confusion_matrix(self, y_pred: np.ndarray) -> None:
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        logger.info("Confusion Matrix → TN=%d | FP=%d | FN=%d | TP=%d", tn, fp, fn, tp)
        detection_lift = (tp / (tp + fn)) / (self.y_test.mean() + 1e-9)
        logger.info("Detection lift vs random: %.1f×", detection_lift)
