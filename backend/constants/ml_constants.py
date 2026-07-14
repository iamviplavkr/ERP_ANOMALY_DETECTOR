"""
backend/constants/ml_constants.py
─────────────────────────────────────────────────────────────────
All domain-specific constants for the ML pipeline.

These values were previously hardcoded across api.py, dashboard.py,
and pipeline.py.  Centralising them here ensures a single point of
change and prevents drift between modules.
"""

# ── PCA / Raw Features ────────────────────────────────────────────────────────

#: The 28 PCA-transformed features from the credit-card dataset
PCA_FEATURES: list[str] = [f"V{i}" for i in range(1, 29)]

#: Engineered features appended during preprocessing
ENGINEERED_FEATURES: list[str] = [
    "log_amount",
    "hour_of_day",
    "is_night",
    "amount_zscore",
]

#: Full ordered feature column list used for model training and inference
ALL_FEATURE_COLS: list[str] = PCA_FEATURES + ENGINEERED_FEATURES

# ── Dataset Statistics (creditcard.csv training split) ────────────────────────
# Used to z-score transaction amounts at inference time so the scaler
# receives the same distribution it saw during training.

AMOUNT_MEAN: float = 88.35   # Mean transaction amount (USD)
AMOUNT_STD: float = 250.12   # Standard deviation

DATASET_TOTAL_TRANSACTIONS: int = 284_807
DATASET_FRAUD_COUNT: int = 492
DATASET_FRAUD_RATE: float = DATASET_FRAUD_COUNT / DATASET_TOTAL_TRANSACTIONS  # ≈ 0.17%

# ── Model Configuration ───────────────────────────────────────────────────────

#: Probability above which a transaction is classified as fraud
DEFAULT_FRAUD_THRESHOLD: float = 0.5

#: Probability above which fraud is considered HIGH risk
DEFAULT_HIGH_RISK_THRESHOLD: float = 0.8

#: Random Forest hyperparameters (must match values used in training)
RF_N_ESTIMATORS: int = 100
RF_CLASS_WEIGHT: str = "balanced"
RF_RANDOM_STATE: int = 42

#: Isolation Forest hyperparameters (used in pipeline.py evaluation)
ISO_CONTAMINATION: float = 0.002
ISO_N_ESTIMATORS: int = 100
ISO_RANDOM_STATE: int = 42

#: Train/test split
TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# ── Risk Level Labels ─────────────────────────────────────────────────────────

RISK_HIGH: str = "HIGH"
RISK_MEDIUM: str = "MEDIUM"
RISK_LOW: str = "LOW"

#: Human-readable alert messages keyed by risk level
RISK_MESSAGES: dict[str, str] = {
    RISK_HIGH:   "⚠️  High-risk transaction flagged. Immediate review required.",
    RISK_MEDIUM: "🔶 Moderate risk detected. Manual verification recommended.",
    RISK_LOW:    "✅ Transaction appears normal. No action required.",
}

# ── Time Constants ────────────────────────────────────────────────────────────

SECONDS_PER_DAY: int = 86_400
NIGHT_HOUR_START: int = 22   # 10 PM
NIGHT_HOUR_END: int = 6      # 6 AM

# ── ERP Context ───────────────────────────────────────────────────────────────

DEPARTMENTS: list[str] = ["Finance", "HR", "Procurement"]
MANAGERS: list[str] = ["mgr_01", "mgr_02", "mgr_03"]

# ── Dashboard ─────────────────────────────────────────────────────────────────

RISK_COLORS: dict[str, str] = {
    RISK_HIGH:   "#ef4444",
    RISK_MEDIUM: "#f59e0b",
    RISK_LOW:    "#10b981",
}

FRAUD_SCORE_BINS = [-0.01, DEFAULT_FRAUD_THRESHOLD, DEFAULT_HIGH_RISK_THRESHOLD, 1.01]
