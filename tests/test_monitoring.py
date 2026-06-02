"""
test_monitoring.py
------------------
Unit tests for the ProductionMonitor class.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.monitoring import ProductionMonitor, PSI_WARNING_THRESHOLD, PSI_ALERT_THRESHOLD


# ── Fixtures ───────────────────────────────────────────────────────────────────

def make_probs(n: int = 1000, seed: int = 42) -> np.ndarray:
    np.random.seed(seed)
    return np.random.beta(2, 5, n)  # realistic default prob distribution


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestProductionMonitor:

    def test_stable_batch_no_alerts(self):
        ref = make_probs(1000, seed=42)
        new = make_probs(1000, seed=43)  # same distribution, different seed
        monitor = ProductionMonitor(reference_probs=ref, reference_auc=0.739)
        report = monitor.run(new, batch_label="2025-Q1")
        assert report["psi_status"] == "STABLE"
        assert len(report["alerts"]) == 0

    def test_drifted_batch_raises_alert(self):
        ref = make_probs(1000, seed=42)
        # Simulate drift: shift distribution significantly
        np.random.seed(99)
        drifted = np.random.beta(5, 2, 1000)  # reversed — much higher probs
        monitor = ProductionMonitor(reference_probs=ref, reference_auc=0.739)
        report = monitor.run(drifted, batch_label="2025-Q2")
        assert report["psi"] >= PSI_ALERT_THRESHOLD
        alert_types = [a["type"] for a in report["alerts"]]
        assert "PSI_HIGH" in alert_types

    def test_auc_drop_triggers_alert(self):
        ref = make_probs(500, seed=42)
        new = make_probs(500, seed=43)
        np.random.seed(1)
        y_true = (np.random.rand(500) < 0.15).astype(int)
        monitor = ProductionMonitor(reference_probs=ref, reference_auc=0.90)
        report = monitor.run(new, y_true=y_true, batch_label="2025-Q3")
        alert_types = [a["type"] for a in report["alerts"]]
        # AUC on random labels will be ~0.5, well below 0.90 reference
        assert "AUC_DROP" in alert_types

    def test_psi_is_zero_for_identical_batches(self):
        ref = make_probs(1000, seed=42)
        monitor = ProductionMonitor(reference_probs=ref, reference_auc=0.739)
        report = monitor.run(ref.copy(), batch_label="identical")
        assert report["psi"] < 0.01

    def test_score_distribution_sums_to_100(self):
        ref = make_probs(1000, seed=42)
        new = make_probs(800, seed=44)
        monitor = ProductionMonitor(reference_probs=ref, reference_auc=0.739)
        report = monitor.run(new)
        total = sum(report["score_distribution"].values())
        assert abs(total - 100.0) < 0.5

    def test_history_accumulates(self):
        ref = make_probs(500, seed=42)
        monitor = ProductionMonitor(reference_probs=ref, reference_auc=0.739)
        for i in range(3):
            monitor.run(make_probs(500, seed=i + 10), batch_label=f"batch_{i}")
        assert len(monitor.history_) == 3

    def test_save_history(self, tmp_path):
        import json
        ref = make_probs(500, seed=42)
        monitor = ProductionMonitor(reference_probs=ref, reference_auc=0.739)
        monitor.run(make_probs(500, seed=10), batch_label="test")
        path = str(tmp_path / "monitoring_log.json")
        monitor.save_history(path)
        with open(path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["batch"] == "test"
