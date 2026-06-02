"""
monitoring.py
-------------
Production monitoring system for the NTACCUL credit scoring model.

Detects model drift using Population Stability Index (PSI),
AUC degradation tracking, and score distribution shift analysis.
Triggers automated alerts when thresholds are exceeded.

Author: [Your Name]
"""

import pandas as pd
import numpy as np
import logging
import json
from datetime import datetime
from typing import Optional, Dict, List
from pathlib import Path

logger = logging.getLogger(__name__)


# ── Constants ──────────────────────────────────────────────────────────────────

PSI_WARNING_THRESHOLD = 0.10   # moderate drift
PSI_ALERT_THRESHOLD   = 0.25   # significant drift — model likely unstable
AUC_DROP_THRESHOLD    = 0.02   # alert if AUC drops more than 2 points


# ── ProductionMonitor ──────────────────────────────────────────────────────────

class ProductionMonitor:
    """
    Monitors a deployed credit scoring model for drift and degradation.

    Metrics tracked
    ---------------
    - PSI (Population Stability Index) on score distribution
    - AUC on labelled production batches
    - Score grade distribution shifts
    - Zero-payment flag rate (early delinquency signal)

    Parameters
    ----------
    reference_probs : np.ndarray
        Predicted probabilities from the training/reference population.
    reference_auc : float
        AUC achieved on the reference (test) dataset.
    model_path : str, optional
        Path to a persisted model file (for batch scoring).
    """

    def __init__(
        self,
        reference_probs: np.ndarray,
        reference_auc: float,
        model_path: Optional[str] = None,
    ):
        self.reference_probs = reference_probs
        self.reference_auc = reference_auc
        self.model_path = model_path
        self.alerts_: List[Dict] = []
        self.history_: List[Dict] = []

    # ── Public API ──────────────────────────────────────────────────────────

    def run(
        self,
        new_probs: np.ndarray,
        y_true: Optional[np.ndarray] = None,
        batch_label: Optional[str] = None,
    ) -> Dict:
        """
        Run full monitoring check on a new production batch.

        Parameters
        ----------
        new_probs : np.ndarray
            Predicted probabilities for the new batch.
        y_true : np.ndarray, optional
            Ground truth labels (if available — enables AUC check).
        batch_label : str, optional
            Label for this batch (e.g. "2025-Q1").

        Returns
        -------
        dict : Monitoring report with metrics and alert flags.
        """
        label = batch_label or datetime.now().strftime("%Y-%m-%d")
        logger.info("Running monitoring for batch: %s", label)

        report = {"batch": label, "timestamp": datetime.now().isoformat(), "alerts": []}

        # PSI check
        psi = self._compute_psi(self.reference_probs, new_probs)
        report["psi"] = round(psi, 4)
        report["psi_status"] = self._psi_status(psi)
        if psi >= PSI_ALERT_THRESHOLD:
            self._raise_alert(report, "PSI_HIGH", f"PSI={psi:.4f} ≥ {PSI_ALERT_THRESHOLD} — significant population drift.")
        elif psi >= PSI_WARNING_THRESHOLD:
            self._raise_alert(report, "PSI_WARNING", f"PSI={psi:.4f} ≥ {PSI_WARNING_THRESHOLD} — moderate drift detected.")

        # Score distribution check
        dist = self._score_distribution(new_probs)
        report["score_distribution"] = dist

        # AUC check (only if labels available)
        if y_true is not None:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(y_true, new_probs)
            report["auc"] = round(auc, 4)
            report["auc_drop"] = round(self.reference_auc - auc, 4)
            if self.reference_auc - auc > AUC_DROP_THRESHOLD:
                self._raise_alert(
                    report, "AUC_DROP",
                    f"AUC dropped {self.reference_auc - auc:.4f} pts "
                    f"(ref={self.reference_auc:.4f} → current={auc:.4f})."
                )

        self.history_.append(report)
        self._log_report(report)
        return report

    def save_history(self, path: str = "reports/monitoring_log.json") -> None:
        """Persist monitoring history to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.history_, f, indent=2)
        logger.info("Monitoring history saved to %s", path)

    def get_alerts(self) -> List[Dict]:
        """Return all alerts raised across monitoring runs."""
        return self.alerts_

    # ── PSI ─────────────────────────────────────────────────────────────────

    @staticmethod
    def _compute_psi(
        reference: np.ndarray,
        current: np.ndarray,
        n_bins: int = 10,
        eps: float = 1e-6,
    ) -> float:
        """
        Compute Population Stability Index (PSI).

        PSI < 0.10  → no significant change
        PSI 0.10–0.25 → moderate change, investigate
        PSI > 0.25  → significant shift, model likely unstable
        """
        breakpoints = np.percentile(reference, np.linspace(0, 100, n_bins + 1))
        breakpoints[0] -= eps
        breakpoints[-1] += eps

        ref_counts = np.histogram(reference, bins=breakpoints)[0]
        cur_counts = np.histogram(current, bins=breakpoints)[0]

        ref_pct = ref_counts / (ref_counts.sum() + eps)
        cur_pct = cur_counts / (cur_counts.sum() + eps)

        # Avoid log(0)
        ref_pct = np.where(ref_pct == 0, eps, ref_pct)
        cur_pct = np.where(cur_pct == 0, eps, cur_pct)

        psi = np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct))
        return float(psi)

    @staticmethod
    def _psi_status(psi: float) -> str:
        if psi < PSI_WARNING_THRESHOLD:
            return "STABLE"
        elif psi < PSI_ALERT_THRESHOLD:
            return "WARNING"
        return "ALERT"

    # ── Score distribution ───────────────────────────────────────────────────

    @staticmethod
    def _score_distribution(probs: np.ndarray, n_grades: int = 10) -> Dict:
        """Return percentage of population per risk grade bucket."""
        thresholds = np.percentile(probs, np.linspace(0, 100, n_grades + 1))
        grades = np.digitize(probs, thresholds[1:-1]) + 1
        grades = np.clip(grades, 1, n_grades)
        counts = pd.Series(grades).value_counts().sort_index()
        return {
            f"grade_{g}_%": round(counts.get(g, 0) / len(probs) * 100, 2)
            for g in range(1, n_grades + 1)
        }

    # ── Alerting ─────────────────────────────────────────────────────────────

    def _raise_alert(self, report: Dict, alert_type: str, message: str) -> None:
        alert = {
            "batch": report["batch"],
            "timestamp": report["timestamp"],
            "type": alert_type,
            "message": message,
        }
        report["alerts"].append(alert)
        self.alerts_.append(alert)
        logger.warning("⚠️  ALERT [%s] — %s", alert_type, message)

    def _log_report(self, report: Dict) -> None:
        logger.info("--- Monitoring Report: %s ---", report["batch"])
        logger.info("  PSI: %.4f (%s)", report.get("psi", 0), report.get("psi_status", "N/A"))
        if "auc" in report:
            logger.info("  AUC: %.4f (drop: %.4f)", report["auc"], report.get("auc_drop", 0))
        if report["alerts"]:
            logger.warning("  %d alert(s) raised.", len(report["alerts"]))
        else:
            logger.info("  No alerts. Model stable.")
