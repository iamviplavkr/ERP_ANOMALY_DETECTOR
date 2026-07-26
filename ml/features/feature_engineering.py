"""
ml/features/feature_engineering.py
─────────────────────────────────────────────────────────────────
Shared feature engineering logic consumed by BOTH the API and the
Streamlit dashboard.  Previously duplicated between api.py and
dashboard.py — now a single source of truth.

Two public functions are exposed:
  - engineer_features_from_request  → for single-transaction API calls
  - engineer_features_from_df       → for batch DataFrame (dashboard)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.constants.ml_constants import (
    AMOUNT_MEAN,
    AMOUNT_STD,
    NIGHT_HOUR_END,
    NIGHT_HOUR_START,
    PCA_FEATURES,
    SECONDS_PER_DAY,
)
from backend.core.exceptions import FeatureEngineeringError
from backend.core.logging import get_logger

logger = get_logger(__name__)


def engineer_features_from_request(txn) -> np.ndarray:
    """
    Build the model's feature vector from a ``TransactionRequest`` object.

    Used by the FastAPI prediction endpoint for single-transaction inference.

    Parameters
    ----------
    txn:
        A ``TransactionRequest`` Pydantic model instance.

    Returns
    -------
    np.ndarray
        Shape ``(1, 32)`` — 28 PCA features + 4 engineered features.

    Raises
    ------
    FeatureEngineeringError
        If any feature computation fails (e.g., NaN inputs).
    """
    try:
        log_amount = float(np.log1p(txn.transaction_amount))
        hour_of_day = int(txn.posting_time % SECONDS_PER_DAY) // 3600
        is_night = int(hour_of_day < NIGHT_HOUR_END or hour_of_day > NIGHT_HOUR_START)
        amount_zscore = float((txn.transaction_amount - AMOUNT_MEAN) / AMOUNT_STD)

        pca_values = [float(getattr(txn, col)) for col in PCA_FEATURES]
        engineered_values = [log_amount, hour_of_day, is_night, amount_zscore]

        feature_vector = np.array(pca_values + engineered_values, dtype=np.float64)

        if np.any(np.isnan(feature_vector)):
            raise FeatureEngineeringError(
                "Feature vector contains NaN values. Check input fields.",
                details={"nan_indices": np.where(np.isnan(feature_vector))[0].tolist()},
            )

        logger.debug(
            "Feature vector built | amount=%.2f log_amount=%.4f hour=%d is_night=%d",
            txn.transaction_amount,
            log_amount,
            hour_of_day,
            is_night,
        )
        return feature_vector.reshape(1, -1)

    except FeatureEngineeringError:
        raise
    except Exception as exc:
        raise FeatureEngineeringError(
            f"Failed to engineer features from request: {exc}"
        ) from exc


def engineer_features_from_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Append engineered features to a raw transaction DataFrame.

    Used by the Streamlit dashboard for batch/CSV-upload analysis.
    Expects columns: ``Time``, ``Amount``, ``V1``–``V28``.

    Parameters
    ----------
    df:
        Raw DataFrame in creditcard.csv format.

    Returns
    -------
    pd.DataFrame
        Original DataFrame with four new columns appended.

    Raises
    ------
    FeatureEngineeringError
        If required columns are missing or computation fails.
    """
    required_cols = {"Time", "Amount"}
    missing = required_cols - set(df.columns)
    if missing:
        raise FeatureEngineeringError(
            f"Input DataFrame is missing required columns: {missing}",
            details={"missing_columns": list(missing)},
        )

    try:
        df = df.copy()
        df["log_amount"] = np.log1p(df["Amount"])
        df["hour_of_day"] = (df["Time"] % SECONDS_PER_DAY) // 3600
        df["is_night"] = (
            (df["hour_of_day"] < NIGHT_HOUR_END) | (df["hour_of_day"] > NIGHT_HOUR_START)
        ).astype(int)
        df["amount_zscore"] = (df["Amount"] - AMOUNT_MEAN) / AMOUNT_STD

        logger.debug("DataFrame feature engineering complete | rows=%d", len(df))
        return df

    except FeatureEngineeringError:
        raise
    except Exception as exc:
        raise FeatureEngineeringError(
            f"Failed to engineer features from DataFrame: {exc}"
        ) from exc
