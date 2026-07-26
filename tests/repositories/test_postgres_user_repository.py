"""
tests/repositories/test_postgres_user_repository.py
─────────────────────────────────────────────────────────────────
Unit tests for PostgresUserRepository using SQLite in-memory.

These tests validate that PostgresUserRepository honours the exact
same Dict[str, Any] contract as InMemoryUserRepository, ensuring
the JWT / RBAC / service layer works identically regardless of
which concrete repository is wired.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.session import Base
from backend.models.role import RoleModel
from backend.models.user import UserModel  # noqa: F401 — registers table
from backend.repositories.postgres_user_repository import PostgresUserRepository
from backend.schemas.user import UserCreate


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def db_session():
    """Create an in-memory SQLite database with all tables and seed roles."""
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed the four standard roles
    for name, desc in [
        ("Admin", "Full system administrator"),
        ("Finance User", "Finance department user"),
        ("Fraud Analyst", "Fraud detection analyst"),
        ("Auditor", "Compliance auditor"),
    ]:
        session.add(RoleModel(name=name, description=desc))
    session.commit()

    yield session
    session.close()
    engine.dispose()


@pytest.fixture
def repo(db_session):
    """Return a PostgresUserRepository bound to the test session."""
    return PostgresUserRepository(db_session)


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestPostgresUserRepository:
    """Verify PostgresUserRepository matches the InMemoryUserRepository contract."""

    def test_create_and_get_by_username(self, repo):
        user = UserCreate(
            username="testuser",
            email="test@example.com",
            role="Admin",
            password="testpass123",
        )
        result = repo.create(user)
        assert result["username"] == "testuser"
        assert result["email"] == "test@example.com"
        assert result["role"] == "Admin"
        assert result["is_active"] is True
        assert "password_hash" in result

        fetched = repo.get_by_username("testuser")
        assert fetched is not None
        assert fetched["username"] == "testuser"
        assert fetched["role"] == "Admin"

    def test_get_by_username_case_insensitive(self, repo):
        repo.create(UserCreate(
            username="CaseUser",
            email="case@test.com",
            role="Auditor",
            password="pass",
        ))
        assert repo.get_by_username("caseuser") is not None
        assert repo.get_by_username("CASEUSER") is not None

    def test_get_by_email(self, repo):
        repo.create(UserCreate(
            username="emailuser",
            email="email@test.com",
            role="Auditor",
            password="pass123",
        ))
        fetched = repo.get_by_email("email@test.com")
        assert fetched is not None
        assert fetched["username"] == "emailuser"

    def test_get_by_email_case_insensitive(self, repo):
        repo.create(UserCreate(
            username="emailcase",
            email="Email@Test.Com",
            role="Admin",
            password="pass123",
        ))
        fetched = repo.get_by_email("email@test.com")
        assert fetched is not None

    def test_get_nonexistent_user(self, repo):
        assert repo.get_by_username("ghost") is None

    def test_get_nonexistent_email(self, repo):
        assert repo.get_by_email("ghost@nowhere.com") is None

    def test_create_duplicate_raises(self, repo):
        user = UserCreate(
            username="dupuser",
            email="dup@test.com",
            role="Admin",
            password="pass123",
        )
        repo.create(user)
        with pytest.raises(ValueError, match="already exists"):
            repo.create(user)

    def test_create_invalid_role_raises(self, repo):
        user = UserCreate(
            username="badrole",
            email="bad@test.com",
            role="NonExistentRole",
            password="pass123",
        )
        with pytest.raises(ValueError, match="not found"):
            repo.create(user)

    def test_password_is_hashed(self, repo):
        import bcrypt
        user = UserCreate(
            username="hashtest",
            email="hash@test.com",
            role="Finance User",
            password="secret_password",
        )
        result = repo.create(user)
        # The stored hash should NOT equal the plain password
        assert result["password_hash"] != "secret_password"
        # But bcrypt.checkpw should verify it
        assert bcrypt.checkpw(
            b"secret_password",
            result["password_hash"].encode("utf-8"),
        )

    def test_dict_contract_matches_inmemory(self, repo):
        """Verify the returned dict has the exact same keys as InMemoryUserRepository."""
        user = UserCreate(
            username="contractuser",
            email="contract@test.com",
            role="Finance User",
            password="pass123",
        )
        result = repo.create(user)
        expected_keys = {"id", "username", "email", "role", "is_active", "password_hash"}
        assert set(result.keys()) == expected_keys

    def test_is_active_default_true(self, repo):
        user = UserCreate(
            username="activeuser",
            email="active@test.com",
            role="Fraud Analyst",
            password="pass123",
        )
        result = repo.create(user)
        assert result["is_active"] is True

    def test_is_active_false(self, repo):
        user = UserCreate(
            username="inactiveuser",
            email="inactive@test.com",
            role="Admin",
            password="pass123",
            is_active=False,
        )
        result = repo.create(user)
        assert result["is_active"] is False

    def test_all_four_roles(self, repo):
        """Verify all four normalised roles can be assigned."""
        roles = ["Admin", "Finance User", "Fraud Analyst", "Auditor"]
        for i, role in enumerate(roles):
            user = UserCreate(
                username=f"user_{i}",
                email=f"user{i}@test.com",
                role=role,
                password="pass",
            )
            result = repo.create(user)
            assert result["role"] == role
