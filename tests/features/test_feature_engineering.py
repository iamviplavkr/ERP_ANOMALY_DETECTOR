"""
tests/features/test_feature_engineering.py
─────────────────────────────────────────────────────────────────
Unit tests for single and DataFrame-level feature engineering.
"""

import pytest
import pandas as pd
import numpy as np
from backend.schemas.transaction import TransactionRequest
from ml.features.feature_engineering import (
    engineer_features_from_request,
    engineer_features_from_df,
)
from backend.core.exceptions import FeatureEngineeringError


def test_engineer_features_from_request():
    txn = TransactionRequest(
        vendor_id="V001",
        department="Finance",
        approved_by="mgr_01",
        posting_time=43200.0,  # 12 hours
        transaction_amount=100.0,
        **{f"V{i}": float(i) for i in range(1, 29)}
    )
    features = engineer_features_from_request(txn)
    assert features.shape == (1, 32)
    # Check night calculation: 12 hours is noon -> is_night should be 0
    assert features[0][30] == 0.0
    # Check log_amount: log1p(100.0)
    assert np.allclose(features[0][28], np.log1p(100.0))


def test_engineer_features_from_df():
    data = {
        "Time": [3600.0, 7200.0],
        "Amount": [100.0, 200.0],
        **{f"V{i}": [0.0, 1.0] for i in range(1, 29)}
    }
    df = pd.DataFrame(data)
    df_feat = engineer_features_from_df(df)
    assert "log_amount" in df_feat.columns
    assert "hour_of_day" in df_feat.columns
    assert "is_night" in df_feat.columns
    assert "amount_zscore" in df_feat.columns
    assert len(df_feat) == 2


def test_engineer_features_from_df_missing_columns():
    df = pd.DataFrame({"Amount": [10.0]})
    with pytest.raises(FeatureEngineeringError):
        engineer_features_from_df(df)
