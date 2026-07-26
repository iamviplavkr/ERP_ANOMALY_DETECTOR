"""
scripts/debug_persistence.py
─────────────────────────────────────────────────────────────────
End-to-end persistence diagnostic for /v1/predict.
Checks DB connectivity, repo types, INSERT emission, and commit success.
Run from the project root:
    venv\\Scripts\\python.exe scripts/debug_persistence.py
"""

import sys
import logging
import uuid

# ── 0. Setup logging to stdout ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    stream=sys.stdout,
)
log = logging.getLogger("persistence_debug")

# ── 1. Settings & environment ─────────────────────────────────────────────────
log.info("=" * 70)
log.info("STEP 1: Settings")
from backend.core.config import settings
log.info("  ENVIRONMENT : %r", settings.ENVIRONMENT)
log.info("  DATABASE_URL: %r", settings.DATABASE_URL)
log.info("  DB_ECHO     : %r", settings.DB_ECHO)

# ── 2. Direct DB connectivity + current database + table list ─────────────────
log.info("=" * 70)
log.info("STEP 2: Direct DB connectivity")
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

try:
    engine = create_engine(settings.DATABASE_URL, echo=True, pool_pre_ping=True)
    with engine.connect() as conn:
        db_name = conn.execute(text("SELECT current_database()")).scalar()
        log.info("  Connected to database: %r", db_name)

        tables = conn.execute(
            text("SELECT tablename FROM pg_tables WHERE schemaname='public' ORDER BY tablename")
        ).fetchall()
        log.info("  Public tables: %s", [t[0] for t in tables])

        for tbl in ["transactions", "predictions", "alerts", "vendors", "audit_logs", "users"]:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                log.info("  %s: %d rows", tbl, count)
            except Exception as exc:
                log.warning("  %s: ERROR - %s", tbl, exc)
except Exception as exc:
    log.error("  DB connection FAILED: %s", exc)
    sys.exit(1)

# ── 3. Session lifecycle test — does flush+commit actually persist? ─────────────
log.info("=" * 70)
log.info("STEP 3: Session lifecycle — manual INSERT test")
from backend.models.transaction import TransactionModel

Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
db = Session()
test_id = uuid.uuid4()
try:
    log.info("  Creating TransactionModel id=%s", test_id)
    tx = TransactionModel(
        id=test_id,
        vendor_id="DEBUG_VENDOR",
        department="Debug",
        approved_by="debug_user",
        posting_time=0.0,
        transaction_amount=9999.0,
        pca_features={"V1": 0.0},
        submitted_by=None,
    )
    db.add(tx)
    log.info("  add() OK")
    db.flush()
    log.info("  flush() OK")
    db.commit()
    log.info("  commit() OK")

    # Verify by reading back in a NEW session
    db2 = Session()
    result = db2.execute(
        text("SELECT vendor_id, transaction_amount FROM transactions WHERE id = :id"),
        {"id": str(test_id)}
    ).fetchone()
    if result:
        log.info("  VERIFIED: row found in DB vendor_id=%s amount=%s", result[0], result[1])
    else:
        log.error("  VERIFICATION FAILED: row NOT found after commit!")
    db2.close()

    # Cleanup
    db3 = Session()
    db3.execute(text("DELETE FROM transactions WHERE id = :id"), {"id": str(test_id)})
    db3.commit()
    db3.close()
    log.info("  Cleanup done")
except Exception as exc:
    log.error("  Session lifecycle test FAILED: %s", exc, exc_info=True)
    db.rollback()
finally:
    db.close()

# ── 4. Simulate DI: what repos does get_db + get_transaction_repository give? ─
log.info("=" * 70)
log.info("STEP 4: DI simulation — what types get injected?")
log.info("  ENVIRONMENT: %r", settings.ENVIRONMENT)

db4 = Session()
log.info("  db type      : %s", type(db4).__name__)
log.info("  db id        : %d", id(db4))

from backend.repositories.transaction_repository import (
    PostgresTransactionRepository,
    InMemoryTransactionRepository,
)
from backend.repositories.prediction_repository import (
    PostgresPredictionRepository,
    InMemoryPredictionRepository,
)

if settings.ENVIRONMENT == "testing":
    tx_repo = InMemoryTransactionRepository()
    pred_repo = InMemoryPredictionRepository()
    log.error("  !! ENVIRONMENT=testing -> InMemory repos -> NO DB WRITES !!")
else:
    tx_repo = PostgresTransactionRepository(db4)
    pred_repo = PostgresPredictionRepository(db4)
    log.info("  transaction_repo type: %s", type(tx_repo).__name__)
    log.info("  prediction_repo  type: %s", type(pred_repo).__name__)
    log.info("  tx_repo.db id        : %d", id(tx_repo.db))
    log.info("  pred_repo.db id      : %d", id(pred_repo.db))
    log.info("  Same session object? : %s", tx_repo.db is pred_repo.db)
db4.close()

# ── 5. Full PredictionService.predict_single() simulation ─────────────────────
log.info("=" * 70)
log.info("STEP 5: Full PredictionService.predict_single() simulation")
from backend.schemas.transaction import TransactionRequest
from backend.services.prediction_service import PredictionService
from backend.repositories.audit_log_repository import (
    PostgresAuditLogRepository,
    InMemoryAuditLogRepository,
)
from backend.repositories.alert_repository import (
    PostgresAlertRepository,
    InMemoryAlertRepository,
)
from backend.repositories.vendor_repository import (
    PostgresVendorRepository,
    InMemoryVendorRepository,
)

db5 = Session()
log.info("  db5 id: %d", id(db5))

if settings.ENVIRONMENT == "testing":
    t_repo = InMemoryTransactionRepository()
    p_repo = InMemoryPredictionRepository()
    al_repo = InMemoryAuditLogRepository()
    a_repo = InMemoryAlertRepository()
    v_repo = InMemoryVendorRepository()
    db_arg = None
    log.error("  ENVIRONMENT=testing -> InMemory repos -> no DB persistence!")
else:
    t_repo = PostgresTransactionRepository(db5)
    p_repo = PostgresPredictionRepository(db5)
    al_repo = PostgresAuditLogRepository(db5)
    a_repo = PostgresAlertRepository(db5)
    v_repo = PostgresVendorRepository(db5)
    db_arg = db5
    log.info("  t_repo type : %s  id(db)=%d", type(t_repo).__name__, id(t_repo.db))
    log.info("  p_repo type : %s  id(db)=%d", type(p_repo).__name__, id(p_repo.db))
    log.info("  al_repo type: %s  id(db)=%d", type(al_repo).__name__, id(al_repo.db))
    log.info("  a_repo type : %s  id(db)=%d", type(a_repo).__name__, id(a_repo.db))
    log.info("  v_repo type : %s  id(db)=%d", type(v_repo).__name__, id(v_repo.db))

svc = PredictionService(
    transaction_repo=t_repo,
    prediction_repo=p_repo,
    audit_log_repo=al_repo,
    alert_repo=a_repo,
    vendor_repo=v_repo,
    db=db_arg,
)
log.info("  svc.db is None: %s", svc.db is None)
log.info("  svc.transaction_repo is None: %s", svc.transaction_repo is None)

# Build a minimal transaction request
txn_data = {
    "vendor_id": "DEBUG_VENDOR_SVC",
    "department": "Finance",
    "approved_by": "debug_mgr",
    "posting_time": 43200.0,
    "transaction_amount": 1234.56,
}
for i in range(1, 29):
    txn_data[f"V{i}"] = 0.0

txn = TransactionRequest(**txn_data)
log.info("  TransactionRequest built OK")
log.info("  Calling svc.predict_single()...")
try:
    result = svc.predict_single(txn, user_id=None, ip_address="127.0.0.1")
    log.info("  predict_single() returned: anomaly_score=%s risk=%s", result.anomaly_score, result.risk_level)
except Exception as exc:
    log.error("  predict_single() RAISED: %s", exc, exc_info=True)

db5.close()

# ── 6. Final DB state check ────────────────────────────────────────────────────
log.info("=" * 70)
log.info("STEP 6: Final DB row counts")
try:
    with engine.connect() as conn:
        for tbl in ["transactions", "predictions", "alerts", "vendors", "audit_logs"]:
            try:
                count = conn.execute(text(f"SELECT COUNT(*) FROM {tbl}")).scalar()
                log.info("  %s: %d rows", tbl, count)
            except Exception as exc:
                log.warning("  %s: ERROR - %s", tbl, exc)
except Exception as exc:
    log.error("  Final check failed: %s", exc)

log.info("=" * 70)
log.info("Diagnostic complete.")
