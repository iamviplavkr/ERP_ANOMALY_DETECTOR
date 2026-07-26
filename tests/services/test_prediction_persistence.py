"""
tests/services/test_prediction_persistence.py
─────────────────────────────────────────────────────────────────
Integration tests verifying transaction, prediction, and audit log
database persistence using PredictionService over SQLite.
"""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from backend.database.session import Base
from backend.models.role import RoleModel
from backend.models.user import UserModel
from backend.repositories.transaction_repository import PostgresTransactionRepository
from backend.repositories.prediction_repository import PostgresPredictionRepository
from backend.repositories.audit_log_repository import PostgresAuditLogRepository
from backend.repositories.postgres_user_repository import PostgresUserRepository
from backend.services.prediction_service import PredictionService
from backend.schemas.transaction import TransactionRequest


@pytest.fixture
def db_setup():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    Session = sessionmaker(bind=engine)
    session = Session()

    # Seed Admin role and admin user so we have foreign key satisfaction
    admin_role = RoleModel(name="Admin", description="Admin role")
    session.add(admin_role)
    session.flush()

    import bcrypt
    pwd_hash = bcrypt.hashpw(b"password123", bcrypt.gensalt()).decode("utf-8")
    admin_user = UserModel(
        id=None,
        username="admin",
        email="admin@company.com",
        password_hash=pwd_hash,
        role_id=admin_role.id,
        is_active=True,
    )
    session.add(admin_user)
    session.commit()
    session.refresh(admin_user)

    user_id = str(admin_user.id)

    yield session, user_id

    session.close()
    engine.dispose()


class TestPredictionPersistence:
    """Verify PredictionService records transactions, predictions, and audits."""

    def test_predict_single_persists_successfully(self, db_setup, mock_model_artifacts):
        session, user_id = db_setup

        tx_repo = PostgresTransactionRepository(session)
        pred_repo = PostgresPredictionRepository(session)
        audit_repo = PostgresAuditLogRepository(session)

        service = PredictionService(
            transaction_repo=tx_repo,
            prediction_repo=pred_repo,
            audit_log_repo=audit_repo,
            db=session,
        )

        txn = TransactionRequest(
            vendor_id="V_PERSIST_TEST",
            department="HR",
            approved_by="mgr_02",
            posting_time=120.0,
            transaction_amount=350.0,
            **{f"V{i}": 0.0 for i in range(1, 29)}
        )

        res = service.predict_single(txn, user_id=user_id, ip_address="10.0.0.5")

        assert res.vendor_id == "V_PERSIST_TEST"
        assert res.transaction_amount == 350.0

        # Query database to confirm transaction record exists
        txs = tx_repo.db.query(tx_repo.db.query(Base.metadata.tables["transactions"]).subquery()).all()
        assert len(txs) == 1
        db_tx = txs[0]
        assert db_tx.vendor_id == "V_PERSIST_TEST"
        assert str(db_tx.submitted_by) == user_id

        # Confirm prediction record exists and links correctly
        preds = pred_repo.db.query(pred_repo.db.query(Base.metadata.tables["predictions"]).subquery()).all()
        assert len(preds) == 1
        db_pred = preds[0]
        assert db_pred.transaction_id == db_tx.id
        assert db_pred.anomaly_score == res.anomaly_score
        assert db_pred.risk_level == res.risk_level

        # Confirm audit log exists
        audits = audit_repo.get_all()
        assert len(audits) == 1
        db_audit = audits[0]
        assert db_audit["action"] == "PREDICT"
        assert db_audit["user_id"] == user_id
        assert db_audit["ip_address"] == "10.0.0.5"

    def test_predict_batch_persists_atomically(self, db_setup, mock_model_artifacts):
        session, user_id = db_setup

        tx_repo = PostgresTransactionRepository(session)
        pred_repo = PostgresPredictionRepository(session)
        audit_repo = PostgresAuditLogRepository(session)

        service = PredictionService(
            transaction_repo=tx_repo,
            prediction_repo=pred_repo,
            audit_log_repo=audit_repo,
            db=session,
        )

        txns = [
            TransactionRequest(
                vendor_id="V_BATCH_1",
                department="Finance",
                approved_by="mgr_01",
                posting_time=100.0,
                transaction_amount=150.0,
                **{f"V{i}": 0.0 for i in range(1, 29)}
            ),
            TransactionRequest(
                vendor_id="V_BATCH_2",
                department="Finance",
                approved_by="mgr_01",
                posting_time=105.0,
                transaction_amount=250.0,
                **{f"V{i}": 0.0 for i in range(1, 29)}
            )
        ]

        res = service.predict_batch(txns, user_id=user_id, ip_address="10.0.0.6")
        assert res["total"] == 2

        # Check transaction records
        txs = tx_repo.db.query(tx_repo.db.query(Base.metadata.tables["transactions"]).subquery()).all()
        assert len(txs) == 2

        # Check prediction records
        preds = pred_repo.db.query(pred_repo.db.query(Base.metadata.tables["predictions"]).subquery()).all()
        assert len(preds) == 2

        # Check audit log contains ONLY the single BATCH_PREDICT audit log
        audits = audit_repo.get_all()
        assert len(audits) == 1
        assert audits[0]["action"] == "BATCH_PREDICT"
        assert audits[0]["user_id"] == user_id
        assert audits[0]["ip_address"] == "10.0.0.6"
        assert audits[0]["details"]["total"] == 2
