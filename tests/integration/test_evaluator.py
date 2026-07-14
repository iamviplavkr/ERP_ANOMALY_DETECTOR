"""
tests/integration/test_evaluator.py
─────────────────────────────────────────────────────────────────
Regression tests for the ModelEvaluator.
Verifies that Isolation Forest predictions (which return {1, -1})
are correctly normalized to binary {0, 1} before evaluation.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from ml.evaluation.evaluator import ModelEvaluator


def test_evaluator_isolation_forest_normalization():
    # 1. Create a dummy IsolationForest model fit on simple data
    X = np.random.randn(20, 5)
    iso = IsolationForest(contamination=0.1, random_state=42)
    iso.fit(X)

    # 2. Define standard binary target labels (0 = Normal, 1 = Fraud)
    y_test = np.random.randint(0, 2, len(X))

    # 3. Instantiate evaluator
    evaluator = ModelEvaluator(feature_cols=["f1", "f2", "f3", "f4", "f5"])

    # 4. Running evaluate should run without throwing a ValueError
    metrics = evaluator.evaluate(iso, X, y_test, model_name="Isolation Forest")

    # Assertions
    assert metrics["model_name"] == "Isolation Forest"
    assert "classification_report" in metrics
    assert "confusion_matrix" in metrics
    
    # Assert that predictions are successfully mapped/normalized to {0, 1} (no -1 remains)
    assert set(metrics["predictions"]).issubset({0, 1})
    assert -1 not in metrics["predictions"]
