"""
backend/repositories/user_repository.py
─────────────────────────────────────────────────────────────────
User storage repository interface and in-memory mock database.

Pre-populates default role-based test users with hashed passwords
using bcrypt for database-free verification.
"""

from abc import ABC, abstractmethod
from typing import Dict, Optional, Any
import bcrypt

from backend.schemas.user import UserCreate


class UserRepositoryInterface(ABC):
    """
    Abstract interface for managing user data access.
    Ready to be subclassed by SQLAlchemy/PostgreSQL repositories.
    """

    @abstractmethod
    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        """Retrieve user dictionary including password_hash by username."""
        pass

    @abstractmethod
    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        """Retrieve user dictionary including password_hash by email."""
        pass

    @abstractmethod
    def create(self, user: UserCreate) -> Dict[str, Any]:
        """Create and store a new user."""
        pass


class InMemoryUserRepository(UserRepositoryInterface):
    """
    Thread-safe in-memory database dictionary implementation.
    Preloaded with default test users.
    """
    _instance = None

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            cls._instance = super(InMemoryUserRepository, cls).__new__(cls, *args, **kwargs)
            cls._instance._users = {}
            cls._instance._prepopulate_users()
        return cls._instance

    def _prepopulate_users(self) -> None:
        """
        Creates default mock role-based accounts with hashed passwords.
        Default password is 'password123' for all accounts.
        """
        roles = {
            "admin": "Admin",
            "finance": "Finance User",
            "analyst": "Fraud Analyst",
            "auditor": "Auditor"
        }

        for username, role in roles.items():
            salt = bcrypt.gensalt()
            pwd_hash = bcrypt.hashpw(b"password123", salt).decode("utf-8")
            self._users[username] = {
                "username": username,
                "email": f"{username}@company.com",
                "role": role,
                "is_active": True,
                "password_hash": pwd_hash
            }

    def get_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        return self._users.get(username.lower())

    def get_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        for user in self._users.values():
            if user["email"].lower() == email.lower():
                return user
        return None

    def create(self, user: UserCreate) -> Dict[str, Any]:
        username_lower = user.username.lower()
        if username_lower in self._users:
            raise ValueError(f"User '{user.username}' already exists.")

        salt = bcrypt.gensalt()
        pwd_hash = bcrypt.hashpw(user.password.encode("utf-8"), salt).decode("utf-8")

        new_user = {
            "username": user.username,
            "email": user.email,
            "role": user.role,
            "is_active": user.is_active if user.is_active is not None else True,
            "password_hash": pwd_hash
        }
        self._users[username_lower] = new_user
        return new_user
