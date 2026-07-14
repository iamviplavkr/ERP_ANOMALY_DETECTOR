"""
ml/evaluation/evaluator.py
─────────────────────────────────────────────────────────────────
Model evaluation utilities extracted from pipeline.py.

``ModelEvaluator`` wraps sklearn metrics and surfaces them as a
structured dict so results can be logged, stored, or returned via
the API without cluttering the training script.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    auc,
    classification_report,
    confusion_matrix,
    precision_recall_curve,
)

from backend.core.logging import get_logger

logger = get_logger(__name__)


class ModelEvaluator:
    """
    Evaluates a trained classifier and collects metrics.

    Parameters
    ----------
    feature_cols:
        Ordered list of feature names (used for labelling SHAP output).
    """

    def __init__(self, feature_cols: list[str]) -> None:
        self.feature_cols = feature_cols

    # ── Public API ────────────────────────────────────────────────────────────

    def evaluate(
        self,
        model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        model_name: str = "Model",
    ) -> dict:
        """
        Run classification evaluation and return a metrics dict.

        Parameters
        ----------
        model:
            A fitted sklearn estimator with ``predict`` and ``predict_proba``.
        X_test:
            Scaled feature matrix (test split).
        y_test:
            True binary labels (0 = normal, 1 = fraud).
        model_name:
            Label used in log messages.

        Returns
        -------
        dict
            Keys: ``classification_report``, ``confusion_matrix``,
            ``pr_auc``, ``predictions``, ``probabilities``.
        """
        logger.info("Evaluating %s ...", model_name)

        preds = model.predict(X_test)
        if type(model).__name__ == "IsolationForest":
            preds = (preds == -1).astype(int)

        report = classification_report(
            y_test, preds, target_names=["Normal", "Fraud"], output_dict=True
        )

        logger.info("\n%s", classification_report(y_test, preds, target_names=["Normal", "Fraud"]))

        cm = confusion_matrix(y_test, preds).tolist()

        # Precision-Recall AUC (more meaningful than ROC-AUC on imbalanced data)
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)[:, 1]
            precision, recall, _ = precision_recall_curve(y_test, proba)
            pr_auc = auc(recall, precision)
        elif type(model).__name__ == "IsolationForest":
            # Use negative decision function: higher scores indicate anomalies
            proba = -model.decision_function(X_test)
            precision, recall, _ = precision_recall_curve(y_test, proba)
            pr_auc = auc(recall, precision)
        else:
            proba = preds.astype(float)
            pr_auc = None

        logger.info("%s | PR-AUC: %.4f", model_name, pr_auc or 0.0)

        return {
            "model_name": model_name,
            "classification_report": report,
            "confusion_matrix": cm,
            "pr_auc": round(float(pr_auc), 4) if pr_auc is not None else None,
            "predictions": preds,
            "probabilities": proba,
        }

    def build_flagged_df(
        self,
        df_test: pd.DataFrame,
        predictions: np.ndarray,
        probabilities: np.ndarray,
    ) -> pd.DataFrame:
        """
        Attach ``anomaly_score`` and ``flagged`` columns to the test DataFrame.

        Parameters
        ----------
        df_test:
            Original (un-scaled) test-split DataFrame.
        predictions:
            Binary array from ``model.predict``.
        probabilities:
            Float array from ``model.predict_proba[:, 1]``.

        Returns
        -------
        pd.DataFrame
            Test DataFrame with new ``anomaly_score`` and ``flagged`` columns.
        """
        result = df_test.reset_index(drop=True).copy()
        result["anomaly_score"] = probabilities
        result["flagged"] = predictions
        return result

    def top_features_by_importance(
        self,
        model,
        n: int = 10,
    ) -> pd.Series:
        """
        Return the top-N feature importances from a Random Forest.

        Parameters
        ----------
        model:
            A fitted ``RandomForestClassifier``.
        n:
            Number of top features to return.

        Returns
        -------
        pd.Series
            Index = feature names, values = mean decrease in impurity.
        """
        importances = pd.Series(
            np.abs(model.feature_importances_),
            index=self.feature_cols,
        )
        top = importances.sort_values(ascending=False).head(n)
        logger.info("Top %d features by importance:\n%s", n, top.to_string())
        return top
