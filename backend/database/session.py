"""
backend/database/session.py
─────────────────────────────────────────────────────────────────
SQLAlchemy engine, session factory, and FastAPI dependency.

Engine creation is lazy — no actual database connection is made until
a query is executed.  In testing mode (ENVIRONMENT=testing) the
get_db() dependency yields None so no real database is required.
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from backend.core.config import settings
from backend.core.logging import get_logger

logger = get_logger(__name__)


# ── Declarative Base ──────────────────────────────────────────────────────────

class Base(DeclarativeBase):
    """Shared declarative base for every ORM model."""
    pass


# ── Lazy engine / session singletons ─────────────────────────────────────────

_engine = None
_session_factory = None


def get_engine():
    """Return (and lazily create) the SQLAlchemy engine.

    Raises ``ConfigurationError`` when *DATABASE_URL* is empty so that
    a misconfigured production deploy fails loudly instead of silently
    falling back to in-memory storage.
    """
    global _engine
    if _engine is None:
        if not settings.DATABASE_URL:
            from backend.core.exceptions import ConfigurationError
            raise ConfigurationError(
                "DATABASE_URL is not configured. "
                "Set DATABASE_URL in your .env or environment variables."
            )
        _engine = create_engine(
            settings.DATABASE_URL,
            echo=settings.DB_ECHO,
            pool_pre_ping=True,
        )
        logger.info("Database engine created.")
    return _engine


def get_session_factory():
    """Return (and lazily create) the bound session factory."""
    global _session_factory
    if _session_factory is None:
        _session_factory = sessionmaker(
            bind=get_engine(),
            autocommit=False,
            autoflush=False,
        )
    return _session_factory


def dispose_engine() -> None:
    """Dispose the engine connection pool (call on shutdown)."""
    global _engine, _session_factory
    if _engine is not None:
        _engine.dispose()
        logger.info("Database engine disposed.")
        _engine = None
        _session_factory = None


# ── FastAPI dependency ────────────────────────────────────────────────────────

def get_db():
    """Yield a scoped SQLAlchemy session for request-lifetime DI.

    In *testing* mode the dependency yields ``None`` so that the auth
    dependency layer can substitute ``InMemoryUserRepository`` without
    ever touching a real database.
    """
    if settings.ENVIRONMENT == "testing":
        yield None
        return

    factory = get_session_factory()
    db: Session = factory()
    try:
        yield db
    finally:
        db.close()
