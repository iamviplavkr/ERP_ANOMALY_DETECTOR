"""
backend/database/seed.py
─────────────────────────────────────────────────────────────────
Idempotent seeding of development data.

Creates the four default roles and four development users
(admin, finance, analyst, auditor) with bcrypt-hashed passwords.
Skips any rows that already exist.
"""

import bcrypt
from sqlalchemy import select
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.role import RoleModel
from backend.models.user import UserModel

logger = get_logger(__name__)

# ── Seed data ─────────────────────────────────────────────────────────────────

_ROLE_DATA = [
    ("Admin", "Full system administrator"),
    ("Finance User", "Finance department user"),
    ("Fraud Analyst", "Fraud detection analyst"),
    ("Auditor", "Compliance auditor"),
]

_USER_DATA = [
    # (username, email, role_name)
    ("admin", "admin@company.com", "Admin"),
    ("finance", "finance@company.com", "Finance User"),
    ("analyst", "analyst@company.com", "Fraud Analyst"),
    ("auditor", "auditor@company.com", "Auditor"),
]

_DEFAULT_PASSWORD = b"password123"


def seed_default_users(engine) -> None:
    """Idempotently seed roles and default development users.

    Safe to call on every startup — existing rows are silently skipped.
    """
    with Session(engine) as session:
        # 1. Seed roles
        role_map: dict[str, int] = {}
        for name, description in _ROLE_DATA:
            role = session.execute(
                select(RoleModel).filter_by(name=name)
            ).scalar_one_or_none()
            if role is None:
                role = RoleModel(name=name, description=description)
                session.add(role)
                session.flush()  # obtain role.id
                logger.info(f"Seeded role: {name}")
            role_map[name] = role.id

        # 2. Seed users
        for username, email, role_name in _USER_DATA:
            existing = session.execute(
                select(UserModel).filter_by(username=username)
            ).scalar_one_or_none()
            if existing is None:
                salt = bcrypt.gensalt()
                pwd_hash = bcrypt.hashpw(_DEFAULT_PASSWORD, salt).decode("utf-8")
                user = UserModel(
                    username=username,
                    email=email,
                    password_hash=pwd_hash,
                    role_id=role_map[role_name],
                    is_active=True,
                )
                session.add(user)
                logger.info(f"Seeded user: {username} ({role_name})")

        session.commit()
        logger.info("Database seeding complete.")
