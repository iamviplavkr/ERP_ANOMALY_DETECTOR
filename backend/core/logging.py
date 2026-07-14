"""
backend/core/logging.py
─────────────────────────────────────────────────────────────────
Centralised logging configuration.

Call `setup_logging()` once at application startup (in `main.py`).
Then use `get_logger(__name__)` in every module to get a properly
configured logger without repeating handler setup.

Features
--------
- Console handler (coloured output in development)
- Rotating file handler (max 10 MB × 5 backups by default)
- ISO-8601 timestamps
- All configuration comes from `settings`
"""

import logging
import logging.handlers
import os
import sys
from pathlib import Path


# Lazy import to avoid circular dependency at parse time
def _get_settings():
    from backend.core.config import settings  # noqa: PLC0415
    return settings


_LOG_FORMAT = (
    "%(asctime)s | %(levelname)-8s | %(name)-40s | %(message)s"
)
_DATE_FORMAT = "%Y-%m-%dT%H:%M:%S"

# Track whether setup has been called so we never double-configure
_configured = False


def setup_logging() -> None:
    """
    Configure the root logger with console + rotating-file handlers.

    This should be called **once** at application startup before any
    module calls `get_logger()`.
    """
    global _configured
    if _configured:
        return

    settings = _get_settings()

    # Resolve numeric log level
    numeric_level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)

    # ── Root logger ───────────────────────────────────────────────────────────
    root = logging.getLogger()
    root.setLevel(numeric_level)

    formatter = logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT)

    # ── Console handler ───────────────────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # ── Rotating file handler ─────────────────────────────────────────────────
    log_path = Path(settings.LOG_FILE)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_path),
        maxBytes=settings.LOG_MAX_BYTES,
        backupCount=settings.LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    # Silence noisy third-party loggers
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _configured = True

    root.info(
        "Logging initialised | level=%s | file=%s",
        settings.LOG_LEVEL,
        log_path,
    )


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Usage::

        from backend.core.logging import get_logger
        logger = get_logger(__name__)
        logger.info("Hello from %s", __name__)
    """
    return logging.getLogger(name)
