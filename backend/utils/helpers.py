"""
backend/utils/helpers.py
─────────────────────────────────────────────────────────────────
Helper functions for risk levels, top risk factors, and basic formatting.
"""

import numpy as np
from backend.core.config import settings
from backend.constants.ml_constants import (
    RISK_HIGH,
    RISK_MEDIUM,
    RISK_LOW,
    RISK_MESSAGES,
)


def get_risk_level(score: float) -> tuple[str, str]:
    """
    Returns risk level label and alert message based on fraud score.
    """
    if score >= settings.HIGH_RISK_THRESHOLD:
        return RISK_HIGH, RISK_MESSAGES[RISK_HIGH]
    elif score >= settings.FRAUD_THRESHOLD:
        return RISK_MEDIUM, RISK_MESSAGES[RISK_MEDIUM]
    else:
        return RISK_LOW, RISK_MESSAGES[RISK_LOW]


def get_top_risk_factors(
    features: np.ndarray,
    score: float,
    scaler,
    model,
    feature_cols: list[str]
) -> list[dict]:
    """
    Identifies top-N risk factors for a given transaction based on feature importance
    and feature values.
    """
    try:
        # Scale the features
        scaled = scaler.transform(features)[0]
        importances = model.feature_importances_
        # Sort feature indices by importance descending
        top_idx = np.argsort(importances)[::-1][:settings.TOP_RISK_FACTORS_N]

        factors = []
        for i in top_idx:
            factors.append({
                "feature":    feature_cols[i],
                "importance": round(float(importances[i]), 4),
                "value":      round(float(features[0][i]), 4)
            })
        return factors
    except Exception:
        # Fallback if prediction explanations fail
        return []
