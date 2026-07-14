"""
backend/core/config.py
─────────────────────────────────────────────────────────────────
Centralised application configuration.

All runtime values are read from environment variables (via .env)
so that **nothing** is hardcoded across the codebase.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the project root (two levels up from this file)
_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_ROOT / ".env", override=False)


class Settings:
    """
    Single source of truth for all configurable values.

    Precedence: environment variable > .env file > default listed here.
    """

    # ── Application ──────────────────────────────────────────────────────────
    APP_NAME: str = os.getenv("APP_NAME", "ERP Anomaly Detector")
    VERSION: str = os.getenv("APP_VERSION", "2.0.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "false").lower() == "true"

    # ── Server ───────────────────────────────────────────────────────────────
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8000"))
    RELOAD: bool = os.getenv("RELOAD", "true").lower() == "true"
    BACKEND_URL: str = os.getenv("BACKEND_URL", "http://127.0.0.1:8000")

    # ── CORS ─────────────────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = os.getenv("CORS_ORIGINS", "*").split(",")

    # ── Model Artifact Paths ─────────────────────────────────────────────────
    # Defaults look in <project_root>/artifacts/ first, then fall back to root.
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH", str(_ROOT / "artifacts" / "model.pkl")
    )
    SCALER_PATH: str = os.getenv(
        "SCALER_PATH", str(_ROOT / "artifacts" / "scaler.pkl")
    )
    FEATURE_COLS_PATH: str = os.getenv(
        "FEATURE_COLS_PATH", str(_ROOT / "artifacts" / "feature_cols.pkl")
    )
    METADATA_PATH: str = os.getenv(
        "METADATA_PATH", str(_ROOT / "artifacts" / "model_metadata.json")
    )

    # ── ML Thresholds & Dataset Statistics ───────────────────────────────────
    FRAUD_THRESHOLD: float = float(os.getenv("FRAUD_THRESHOLD", "0.5"))
    HIGH_RISK_THRESHOLD: float = float(os.getenv("HIGH_RISK_THRESHOLD", "0.8"))

    # Dataset-derived statistics used during feature engineering.
    # These come from the training distribution (creditcard.csv).
    AMOUNT_MEAN: float = float(os.getenv("AMOUNT_MEAN", "88.35"))
    AMOUNT_STD: float = float(os.getenv("AMOUNT_STD", "250.12"))

    # ── Logging ───────────────────────────────────────────────────────────────
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO")
    LOG_FILE: str = os.getenv("LOG_FILE", str(_ROOT / "logs" / "app.log"))
    LOG_MAX_BYTES: int = int(os.getenv("LOG_MAX_BYTES", str(10 * 1024 * 1024)))  # 10 MB
    LOG_BACKUP_COUNT: int = int(os.getenv("LOG_BACKUP_COUNT", "5"))

    # ── Data ─────────────────────────────────────────────────────────────────
    DATA_PATH: str = os.getenv(
        "DATA_PATH", str(_ROOT / "data" / "creditcard.csv")
    )

    # ── Authentication (optional) ─────────────────────────────────────────────
    API_KEY: str = os.getenv("API_KEY", "")
    REQUIRE_API_KEY: bool = os.getenv("REQUIRE_API_KEY", "false").lower() == "true"

    # ── JWT Security Settings ──────────────────────────────────────────────────
    JWT_SECRET_KEY: str = os.getenv("JWT_SECRET_KEY", "super_secret_erp_access_key")
    JWT_REFRESH_SECRET_KEY: str = os.getenv("JWT_REFRESH_SECRET_KEY", "super_secret_erp_refresh_key")
    JWT_ALGORITHM: str = os.getenv("JWT_ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "15"))
    REFRESH_TOKEN_EXPIRE_DAYS: int = int(os.getenv("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

    # ── Database (PostgreSQL) ─────────────────────────────────────────────────
    DATABASE_URL: str = os.getenv("DATABASE_URL", "")
    TEST_DATABASE_URL: str = os.getenv("TEST_DATABASE_URL", "")
    DB_ECHO: bool = os.getenv("DB_ECHO", "false").lower() == "true"

    # ── Flags ─────────────────────────────────────────────────────────────────
    TOP_RISK_FACTORS_N: int = int(os.getenv("TOP_RISK_FACTORS_N", "5"))


# Module-level singleton — import this everywhere
settings = Settings()
