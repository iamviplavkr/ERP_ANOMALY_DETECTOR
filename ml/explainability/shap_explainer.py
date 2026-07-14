"""
ml/explainability/shap_explainer.py
─────────────────────────────────────────────────────────────────
SHAP-based explainability wrapper extracted from pipeline.py.

``ShapExplainer`` wraps a SHAP ``TreeExplainer`` and exposes
helper methods for computing feature importance summaries used
in both the pipeline evaluation and the API's top-risk-factor
response field.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

from backend.core.logging import get_logger

logger = get_logger(__name__)


class ShapExplainer:
    """
    SHAP TreeExplainer wrapper for Random Forest models.

    Parameters
    ----------
    model:
        A fitted ``RandomForestClassifier`` (or any tree ensemble).
    feature_cols:
        Ordered list of feature names matching the model's training columns.
    """

    def __init__(self, model, feature_cols: list[str]) -> None:
        self.feature_cols = feature_cols
        logger.info("Initialising SHAP TreeExplainer ...")
        self.explainer = shap.TreeExplainer(model)
        logger.info("SHAP TreeExplainer ready.")

    # ── Public API ────────────────────────────────────────────────────────────

    def compute_shap_values(
        self,
        X_sample: np.ndarray,
        class_index: int = 1,
    ) -> np.ndarray:
        """
        Compute SHAP values for a sample of observations.

        Parameters
        ----------
        X_sample:
            Feature matrix (scaled), shape ``(n_samples, n_features)``.
        class_index:
            Class to extract values for (1 = fraud class).

        Returns
        -------
        np.ndarray
            SHAP values array, shape ``(n_samples, n_features)``.
        """
        logger.info("Computing SHAP values for %d samples ...", len(X_sample))
        raw = self.explainer.shap_values(X_sample)

        # Handle both old (list) and new (3-D array) SHAP output formats
        if isinstance(raw, list):
            sv = raw[class_index]
        else:
            sv = raw[:, :, class_index]

        logger.info("SHAP computation complete.")
        return sv

    def feature_importance_summary(
        self,
        shap_values: np.ndarray,
        n: int = 10,
    ) -> pd.Series:
        """
        Compute mean absolute SHAP values as a feature importance ranking.

        Parameters
        ----------
        shap_values:
            Output of ``compute_shap_values``.
        n:
            Number of top features to return.

        Returns
        -------
        pd.Series
            Sorted feature importance Series (descending).
        """
        importance = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=self.feature_cols,
        )
        top = importance.sort_values(ascending=False).head(n)
        logger.info("Top %d SHAP features:\n%s", n, top.to_string())
        return top

    def sample_and_explain(
        self,
        X_test: np.ndarray,
        n_samples: int = 300,
        random_state: int = 42,
        top_n: int = 10,
    ) -> pd.Series:
        """
        Convenience method: sample rows, compute SHAP, return importance.

        Parameters
        ----------
        X_test:
            Full test feature matrix.
        n_samples:
            Number of random rows to use (controls compute time).
        random_state:
            Seed for reproducible sampling.
        top_n:
            Number of top features to surface.

        Returns
        -------
        pd.Series
            Feature importance sorted descending.
        """
        rng = np.random.RandomState(random_state)
        n = min(n_samples, len(X_test))
        idx = rng.choice(len(X_test), n, replace=False)

        sv = self.compute_shap_values(X_test[idx])
        return self.feature_importance_summary(sv, n=top_n)
