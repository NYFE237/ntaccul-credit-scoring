"""
export_artifacts.py
-------------------
Train the NTACCUL credit scoring model and export everything the API needs.

Usage
-----
    python -m src.export_artifacts --data data/raw/loan_data.csv --out models

Outputs (in --out)
------------------
credit_scorer.pkl      trained model (same file ``ProductionMonitor`` expects)
preprocessing.json     frozen training statistics + scorecard thresholds
reference_probs.npy    test-set probabilities (reference for PSI monitoring)
metadata.json          model name, metrics, library versions, training date
"""

import argparse
import json
import logging
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import sklearn

from src.data_processing import DataPipeline, NUMERIC_COLUMNS, CATEGORICAL_COLUMNS
from src.feature_engineering import FeatureEngineer
from src.model import CreditScorer
from src.scorecard import Scorecard
from src.serving import DUMMY_COLUMNS, InferencePreprocessor

logger = logging.getLogger(__name__)


class _StatsPipeline(DataPipeline):
    """DataPipeline that records the statistics it uses, so they can be reused at inference."""

    def _clip_outliers(self, lower_q: float = 0.01, upper_q: float = 0.99) -> None:
        # Called after imputation and type enforcement, right before clipping:
        # exactly the moment where the bounds are computed in DataPipeline.
        self.bounds_ = {
            c: [float(self.df[c].quantile(lower_q)), float(self.df[c].quantile(upper_q))]
            for c in NUMERIC_COLUMNS if c in self.df.columns
        }
        # Imputing with the median leaves the median unchanged, so these equal
        # the values DataPipeline used.
        self.medians_ = {c: float(self.df[c].median()) for c in NUMERIC_COLUMNS}
        self.modes_ = {c: str(self.df[c].mode()[0]) for c in CATEGORICAL_COLUMNS}
        super()._clip_outliers(lower_q, upper_q)


def export_artifacts(data_path: str, out_dir: str = "models") -> Dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Clean (and record statistics)
    pipeline = _StatsPipeline(data_path)
    df_clean = pipeline.run()

    # 2. Features
    fe = FeatureEngineer(df_clean)
    df_features = fe.transform()
    feature_columns = fe.get_feature_names()

    # 3. Train / evaluate / persist model
    scorer = CreditScorer(df_features)
    scorer.train()
    metrics = scorer.evaluate()
    scorer.save(str(out / "credit_scorer.pkl"))

    # 4. Scorecard (grades 1-10 calibrated on training probabilities)
    scorecard = Scorecard().fit(
        scorer.predict_proba(scorer.X_train), scorer.y_train.to_numpy()
    )

    # 5. Reference distribution for monitoring
    ref_probs = scorer.predict_proba(scorer.X_test)
    np.save(out / "reference_probs.npy", ref_probs)

    # 6. Frozen preprocessing state
    amount_edges = pd.qcut(
        df_clean["amount_fcfa"], q=5, retbins=True, duplicates="drop"
    )[1]
    state = {
        "bounds": pipeline.bounds_,
        "medians": pipeline.medians_,
        "modes": pipeline.modes_,
        "amount_edges": [float(x) for x in amount_edges],
        "categories": {c: sorted(df_clean[c].unique().tolist()) for c in DUMMY_COLUMNS},
        "feature_columns": list(feature_columns),
        "scorecard_thresholds": [float(x) for x in scorecard.thresholds_],
    }
    InferencePreprocessor(state).save(out / "preprocessing.json")

    # 7. Metadata
    metadata = {
        "model_name": scorer.best_model_name_,
        "benchmark_auc": scorer.benchmark_results_,
        "test_metrics": metrics,
        "n_features": len(feature_columns),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    with open(out / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Artifacts written to %s", out.resolve())
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data/raw/loan_data.csv")
    parser.add_argument("--out", default="models")
    args = parser.parse_args()
    meta = export_artifacts(args.data, args.out)
    print(json.dumps(meta, indent=2))
