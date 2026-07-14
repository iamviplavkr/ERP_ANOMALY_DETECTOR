"""
backend/repositories/model_repository.py
─────────────────────────────────────────────────────────────────
Singleton repository class to load and access model artifacts.
Handles loading model.pkl, scaler.pkl, and feature_cols.pkl,
and throws custom exceptions on failure.
"""

import os
import pickle
from typing import Any

from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.exceptions import ArtifactNotFoundError, ModelLoadError

logger = get_logger(__name__)


class ModelRepository:
    """
    Repository class responsible for managing model artifacts in memory.
    Implements a thread-safe singleton-like behavior by loading assets lazily and caching them.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(ModelRepository, cls).__new__(cls, *args, **kwargs)
            cls._instance._model = None
            cls._instance._scaler = None
            cls._instance._feature_cols = None
            cls._instance._loaded = False
        return cls._instance

    def load_artifacts(self, force: bool = False) -> None:
        """
        Loads the model, scaler, and feature column names from disk.
        """
        if self._loaded and not force:
            return

        # Check model file
        if not os.path.exists(settings.MODEL_PATH):
            logger.error(f"Model file not found at {settings.MODEL_PATH}")
            raise ArtifactNotFoundError(f"Model file not found at {settings.MODEL_PATH}")

        # Check scaler file
        if not os.path.exists(settings.SCALER_PATH):
            logger.error(f"Scaler file not found at {settings.SCALER_PATH}")
            raise ArtifactNotFoundError(f"Scaler file not found at {settings.SCALER_PATH}")

        # Check feature cols file
        if not os.path.exists(settings.FEATURE_COLS_PATH):
            logger.error(f"Feature columns file not found at {settings.FEATURE_COLS_PATH}")
            raise ArtifactNotFoundError(f"Feature columns file not found at {settings.FEATURE_COLS_PATH}")

        try:
            logger.info(f"Loading model from {settings.MODEL_PATH}...")
            with open(settings.MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)

            logger.info(f"Loading scaler from {settings.SCALER_PATH}...")
            with open(settings.SCALER_PATH, "rb") as f:
                self._scaler = pickle.load(f)

            logger.info(f"Loading feature columns from {settings.FEATURE_COLS_PATH}...")
            with open(settings.FEATURE_COLS_PATH, "rb") as f:
                self._feature_cols = pickle.load(f)

            self._loaded = True
            logger.info("All model artifacts loaded successfully.")

        except Exception as e:
            logger.exception("Failed to deserialize model artifacts.")
            raise ModelLoadError(f"Error loading model artifacts: {str(e)}") from e

    @property
    def model(self) -> Any:
        if not self._loaded:
            self.load_artifacts()
        return self._model

    @property
    def scaler(self) -> Any:
        if not self._loaded:
            self.load_artifacts()
        return self._scaler

    @property
    def feature_cols(self) -> list[str]:
        if not self._loaded:
            self.load_artifacts()
        return self._feature_cols

    @property
    def is_loaded(self) -> bool:
        return self._loaded
