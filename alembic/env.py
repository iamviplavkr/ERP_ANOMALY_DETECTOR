"""
alembic/env.py
─────────────────────────────────────────────────────────────────
Alembic migration environment.

Reads DATABASE_URL from application settings so that the connection
string never needs to be duplicated in alembic.ini.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

import os
import sys
from pathlib import Path

# Add project root directory to sys.path to allow imports from backend
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.core.config import settings
from backend.database.session import Base
import backend.models  # noqa: F401 — registers all tables

# ── Alembic Config object ────────────────────────────────────────────────────
config = context.config

# Override sqlalchemy.url from application settings
if settings.DATABASE_URL:
    config.set_main_option("sqlalchemy.url", settings.DATABASE_URL)

# Interpret the config file for Python logging (if present)
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# The MetaData object for 'autogenerate' support
target_metadata = Base.metadata


# ── Offline mode ─────────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL without DB connection)."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online mode ──────────────────────────────────────────────────────────────

def run_migrations_online() -> None:
    """Run migrations in 'online' mode (real DB connection)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
