"""
backend/models — SQLAlchemy ORM models.

Importing this package registers every model with ``Base.metadata``
so that Alembic auto-generation and ``create_all`` see all tables.
"""

from backend.models.role import RoleModel  # noqa: F401
from backend.models.user import UserModel  # noqa: F401
from backend.models.transaction import TransactionModel  # noqa: F401
from backend.models.prediction import PredictionModel  # noqa: F401
from backend.models.audit_log import AuditLogModel  # noqa: F401
from backend.models.alert import AlertModel  # noqa: F401
from backend.models.vendor import VendorRiskModel  # noqa: F401

