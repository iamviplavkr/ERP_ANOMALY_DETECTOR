"""
ml/training/train.py
─────────────────────────────────────────────────────────────────
Trains the Random Forest model on ERP transaction data (creditcard.csv)
and saves the model.pkl, scaler.pkl, and feature_cols.pkl artifacts.

Replaces the legacy save_model.py with robust logging, path configurations
from settings, and proper exception handling.
"""

import os
import pickle
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from backend.core.config import settings
from backend.core.logging import get_logger, setup_logging
from backend.constants.ml_constants import (
    ALL_FEATURE_COLS,
    RF_N_ESTIMATORS,
    RF_CLASS_WEIGHT,
    RF_RANDOM_STATE,
    TEST_SIZE,
    RANDOM_STATE,
)
from ml.features.feature_engineering import engineer_features_from_df

warnings.filterwarnings('ignore')
setup_logging()
logger = get_logger(__name__)


def train_pipeline() -> None:
    """
    Executes the full training pipeline:
      1. Loads raw dataset from settings.DATA_PATH
      2. Renames variables to ERP context
      3. Performs feature engineering
      4. Standardizes the features
      5. Trains a Random Forest classifier
      6. Serializes model artifacts to backend/config-specified locations
    """
    logger.info("Starting training pipeline...")
    data_path = settings.DATA_PATH
    if not os.path.exists(data_path):
        logger.error(f"Dataset not found at {data_path}. Please download creditcard.csv and place it in the data/ directory.")
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    logger.info(f"Loading dataset from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.rename(columns={
        'Time':   'posting_time',
        'Amount': 'transaction_amount',
        'Class':  'is_fraud'
    })

    # Synthetic ERP contexts
    np.random.seed(42)
    df['department']  = np.random.choice(['Finance', 'HR', 'Procurement'], len(df))
    df['vendor_id']   = 'V' + df.index.astype(str).str.zfill(5)
    df['approved_by'] = np.random.choice(['mgr_01', 'mgr_02', 'mgr_03'], len(df))

    logger.info("Performing feature engineering...")
    # Map column names expected by engineer_features_from_df (Time and Amount)
    df_temp = df.rename(columns={
        'posting_time': 'Time',
        'transaction_amount': 'Amount'
    })
    df_feat = engineer_features_from_df(df_temp)
    # Rename back or retrieve engineered values
    df['log_amount'] = df_feat['log_amount']
    df['hour_of_day'] = df_feat['hour_of_day']
    df['is_night'] = df_feat['is_night']
    df['amount_zscore'] = df_feat['amount_zscore']

    feature_cols = ALL_FEATURE_COLS
    logger.info(f"Using {len(feature_cols)} features for training.")

    X = df[feature_cols].values
    y = df['is_fraud'].values

    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_sc, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    logger.info("Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    logger.info("Model training complete.")

    # Save artifacts (ensure directories exist)
    for path_str in [settings.MODEL_PATH, settings.SCALER_PATH, settings.FEATURE_COLS_PATH]:
        os.makedirs(os.path.dirname(path_str), exist_ok=True)

    logger.info(f"Saving model to {settings.MODEL_PATH}...")
    with open(settings.MODEL_PATH, 'wb') as f:
        pickle.dump(rf, f)

    logger.info(f"Saving scaler to {settings.SCALER_PATH}...")
    with open(settings.SCALER_PATH, 'wb') as f:
        pickle.dump(scaler, f)

    logger.info(f"Saving feature columns to {settings.FEATURE_COLS_PATH}...")
    with open(settings.FEATURE_COLS_PATH, 'wb') as f:
        pickle.dump(feature_cols, f)

    logger.info("Training pipeline finished successfully. All artifacts saved.")


if __name__ == "__main__":
    train_pipeline()
