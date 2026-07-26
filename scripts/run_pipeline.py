"""
scripts/run_pipeline.py
─────────────────────────────────────────────────────────────────
Runs the complete ML pipeline: loads dataset, fits models, runs
SHAP analysis, evaluates metrics, and outputs results.

Uses the refactored ml.evaluation and ml.explainability modules.
"""

import os
import sys
import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Ensure project root is in python path
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.config import settings
from backend.core.logging import get_logger, setup_logging
from backend.constants.ml_constants import (
    ALL_FEATURE_COLS,
    TEST_SIZE,
    RANDOM_STATE,
    ISO_CONTAMINATION,
    ISO_N_ESTIMATORS,
    ISO_RANDOM_STATE,
    RF_N_ESTIMATORS,
    RF_CLASS_WEIGHT,
    RF_RANDOM_STATE,
)
from ml.features.feature_engineering import engineer_features_from_df
from ml.evaluation.evaluator import ModelEvaluator
from ml.explainability.shap_explainer import ShapExplainer

warnings.filterwarnings('ignore')
setup_logging()
logger = get_logger(__name__)


def main():
    logger.info("Initializing pipeline run...")

    data_path = settings.DATA_PATH
    if not os.path.exists(data_path):
        logger.error(f"Dataset creditcard.csv not found at {data_path}")
        print(f"Error: Dataset not found. Please download creditcard.csv and place it at {data_path}")
        sys.exit(1)

    logger.info(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    df = df.rename(columns={
        'Time':   'posting_time',
        'Amount': 'transaction_amount',
        'Class':  'is_fraud'
    })

    # Add synthetic contexts
    np.random.seed(42)
    df['department']  = np.random.choice(['Finance', 'HR', 'Procurement'], len(df))
    df['vendor_id']   = 'V' + df.index.astype(str).str.zfill(5)
    df['approved_by'] = np.random.choice(['mgr_01', 'mgr_02', 'mgr_03'], len(df))

    logger.info("Running feature engineering...")
    df_temp = df.rename(columns={
        'posting_time': 'Time',
        'transaction_amount': 'Amount'
    })
    df_feat = engineer_features_from_df(df_temp)
    df['log_amount'] = df_feat['log_amount']
    df['hour_of_day'] = df_feat['hour_of_day']
    df['is_night'] = df_feat['is_night']
    df['amount_zscore'] = df_feat['amount_zscore']

    feature_cols = ALL_FEATURE_COLS
    X = df[feature_cols].values
    y = df['is_fraud'].values

    logger.info("Scaling features...")
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    indices = np.arange(len(df))
    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X_sc, y, indices, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Unsupervised Baseline
    logger.info("Fitting Isolation Forest...")
    iso = IsolationForest(
        contamination=ISO_CONTAMINATION,
        random_state=ISO_RANDOM_STATE,
        n_estimators=ISO_N_ESTIMATORS
    )
    iso.fit(X_train)
    iso_preds = (iso.predict(X_test) == -1).astype(int)

    # Supervised Classifier
    logger.info("Fitting Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)

    # Save trained artifacts to backend model repository location
    import pickle
    logger.info(f"Saving model to {settings.MODEL_PATH}...")
    with open(settings.MODEL_PATH, "wb") as f:
        pickle.dump(rf, f)

    logger.info(f"Saving scaler to {settings.SCALER_PATH}...")
    with open(settings.SCALER_PATH, "wb") as f:
        pickle.dump(scaler, f)

    logger.info(f"Saving feature columns to {settings.FEATURE_COLS_PATH}...")
    with open(settings.FEATURE_COLS_PATH, "wb") as f:
        pickle.dump(feature_cols, f)

    # Evaluation
    evaluator = ModelEvaluator(feature_cols=feature_cols)

    print("\n===== Isolation Forest (Unsupervised Baseline) =====")
    evaluator.evaluate(iso, X_test, y_test, model_name="Isolation Forest")

    print("\n===== Random Forest (Supervised Model) =====")
    rf_eval = evaluator.evaluate(rf, X_test, y_test, model_name="Random Forest")
    rf_preds = rf_eval["predictions"]
    rf_proba = rf_eval["probabilities"]

    # Save evaluation metrics metadata
    import json
    metadata = {
        "model_name": "Random Forest",
        "precision": round(float(rf_eval["classification_report"]["Fraud"]["precision"]), 4),
        "recall": round(float(rf_eval["classification_report"]["Fraud"]["recall"]), 4),
        "pr_auc": round(float(rf_eval["pr_auc"]), 4),
        "training_samples": int(len(df))
    }
    metadata_dir = os.path.dirname(settings.METADATA_PATH)
    if metadata_dir:
        os.makedirs(metadata_dir, exist_ok=True)
    with open(settings.METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved evaluation metrics to {settings.METADATA_PATH}")

    # SHAP Explanations
    logger.info("Generating SHAP explanations...")
    shap_explainer = ShapExplainer(rf, feature_cols)
    shap_explainer.sample_and_explain(X_test, n_samples=300, random_state=42, top_n=10)

    # Flagged transactions report
    logger.info("Saving flagged transactions report...")
    df_test = df.iloc[idx_test].reset_index(drop=True).copy()
    flagged_df = evaluator.build_flagged_df(df_test, rf_preds, rf_proba)

    flagged_subset = flagged_df[flagged_df['flagged'] == 1][
        ['vendor_id', 'department', 'approved_by',
         'transaction_amount', 'anomaly_score', 'is_fraud']
    ].head(10)

    print("\nSample flagged transactions:")
    print(flagged_subset.to_string(index=False))

    output_csv = "flagged_transactions.csv"
    flagged_df[flagged_df['flagged'] == 1].to_csv(output_csv, index=False)
    logger.info(f"Flagged transactions saved to {output_csv}")


if __name__ == "__main__":
    main()
