"""
backend/services/prediction_service.py
─────────────────────────────────────────────────────────────────
Service layer implementation for managing single and batch predictions.
Decouples FastAPI handlers from business logic and database/repository access.
Now integrates PostgreSQL persistence for transactions, predictions, and audits.
"""

from typing import Dict, Any, List, Optional
from backend.core.exceptions import PredictionError, BatchPredictionError
from backend.core.logging import get_logger
from backend.repositories.model_repository import ModelRepository
from backend.schemas.transaction import TransactionRequest, TransactionResponse
from ml.features.feature_engineering import engineer_features_from_request
from backend.utils.helpers import get_risk_level, get_top_risk_factors
from backend.services.risk_engine import evaluate_risk

logger = get_logger(__name__)


class PredictionService:
    """
    Main business service class for running ERP anomaly detection.
    Supports optional repository injects to automatically persist records
    and feed the risk intelligence engine.
    """

    def __init__(
        self,
        repository: ModelRepository = None,
        transaction_repo: Any = None,
        prediction_repo: Any = None,
        audit_log_repo: Any = None,
        alert_repo: Any = None,
        vendor_repo: Any = None,
        db: Any = None,
    ) -> None:
        self.repo = repository or ModelRepository()
        self.transaction_repo = transaction_repo
        self.prediction_repo = prediction_repo
        self.audit_log_repo = audit_log_repo
        self.alert_repo = alert_repo
        self.vendor_repo = vendor_repo
        self.db = db

    def predict_single(
        self,
        txn: TransactionRequest,
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        disable_audit: bool = False,
    ) -> TransactionResponse:
        """
        Processes, scores, and optionally persists a single ERP transaction.
        """
        try:
            logger.debug(f"Processing transaction for vendor: {txn.vendor_id}")
            # Ensure model and scaler are loaded
            self.repo.load_artifacts()

            # 1. Feature Engineering
            features = engineer_features_from_request(txn)

            # 2. Standardize/Scale features
            features_sc = self.repo.scaler.transform(features)

            # 3. Model Inference
            proba = self.repo.model.predict_proba(features_sc)[0][1]
            is_fraud = bool(proba >= 0.5)

            # 4. Human-readable Risk Levels & SHAP-style importances
            risk_level, alert_message = get_risk_level(proba)
            top_factors = get_top_risk_factors(
                features, proba, self.repo.scaler, self.repo.model, self.repo.feature_cols
            )

            response = TransactionResponse(
                vendor_id=txn.vendor_id or "UNKNOWN",
                department=txn.department or "Finance",
                approved_by=txn.approved_by or "mgr_01",
                transaction_amount=txn.transaction_amount,
                anomaly_score=round(float(proba), 4),
                is_fraud=is_fraud,
                risk_level=risk_level,
                alert_message=alert_message,
                top_risk_factors=top_factors
            )

            # 5. Database Persistence
            tx_id = None
            if self.transaction_repo:
                pca_features = {
                    f"V{i}": getattr(txn, f"V{i}") for i in range(1, 29)
                }
                tx_data = {
                    "vendor_id": txn.vendor_id,
                    "department": txn.department,
                    "approved_by": txn.approved_by,
                    "posting_time": txn.posting_time,
                    "transaction_amount": txn.transaction_amount,
                    "pca_features": pca_features,
                    "submitted_by": user_id,
                }
                db_tx = self.transaction_repo.create(tx_data)
                tx_id = db_tx["id"]

            if self.prediction_repo and tx_id:
                from backend.core.config import settings
                pred_data = {
                    "transaction_id": tx_id,
                    "anomaly_score": response.anomaly_score,
                    "is_fraud": response.is_fraud,
                    "risk_level": response.risk_level,
                    "alert_message": response.alert_message,
                    "top_risk_factors": response.top_risk_factors,
                    "model_version": settings.VERSION,
                }
                db_pred = self.prediction_repo.create(pred_data)
                pred_id = db_pred["id"]
            else:
                db_pred = None
                pred_id = None

            # 6. Risk Engine Evaluation — generate alerts for HIGH/CRITICAL findings
            if self.alert_repo and pred_id:
                vendor_risk = None
                if self.vendor_repo and txn.vendor_id:
                    try:
                        from backend.services.vendor_service import VendorService
                        vendor_service = VendorService(self.vendor_repo)
                        vendor_risk = vendor_service.get_or_create_default(txn.vendor_id)
                    except Exception as exc:
                        logger.warning(f"Failed to lookup or create vendor risk profile: {exc}")

                tx_context = {
                    "vendor_id": txn.vendor_id,
                    "department": txn.department,
                    "transaction_amount": txn.transaction_amount,
                }
                alert_risk_level, rules_triggered, mitigation = evaluate_risk(
                    transaction_data=tx_context,
                    anomaly_score=response.anomaly_score,
                    top_risk_factors=response.top_risk_factors or [],
                    vendor_risk=vendor_risk,
                )
                if alert_risk_level in ("HIGH", "CRITICAL"):
                    self.alert_repo.create({
                        "prediction_id": pred_id,
                        "risk_level": alert_risk_level,
                        "status": "OPEN",
                        "rules_triggered": rules_triggered,
                        "mitigation_action": mitigation,
                    })
                    logger.info(f"Alert created: level={alert_risk_level}, rules={rules_triggered}")

                # Update vendor stats if a vendor profile is associated
                if self.vendor_repo and vendor_risk and txn.vendor_id:
                    try:
                        from datetime import datetime, timezone
                        new_total = vendor_risk.get("total_transactions_count", 0) + 1
                        updates = {
                            "total_transactions_count": new_total,
                            "last_transaction_at": datetime.now(timezone.utc),
                        }
                        if alert_risk_level in ("HIGH", "CRITICAL"):
                            updates["historical_alerts_count"] = vendor_risk.get("historical_alerts_count", 0) + 1
                            updates["last_alert_at"] = datetime.now(timezone.utc)

                        # Update historical fraud rate
                        old_total = vendor_risk.get("total_transactions_count", 0)
                        old_rate = vendor_risk.get("historical_fraud_rate", 0.0)
                        old_fraud_cnt = round(old_rate * old_total)
                        new_fraud_cnt = old_fraud_cnt + (1 if is_fraud else 0)
                        updates["historical_fraud_rate"] = new_fraud_cnt / new_total

                        self.vendor_repo.update(txn.vendor_id, updates)
                    except Exception as exc:
                        logger.warning(f"Failed to update vendor risk metrics: {exc}")

            # 7. Audit Logging
            if self.audit_log_repo and not disable_audit:
                audit_data = {
                    "user_id": user_id,
                    "action": "PREDICT",
                    "resource": "/v1/predict",
                    "details": {
                        "vendor_id": txn.vendor_id,
                        "transaction_amount": txn.transaction_amount,
                        "anomaly_score": response.anomaly_score,
                        "is_fraud": response.is_fraud,
                    },
                    "ip_address": ip_address,
                }
                self.audit_log_repo.create(audit_data)

            if self.db:
                self.db.commit()

            logger.info(f"Transaction scored and saved: vendor={txn.vendor_id}, score={response.anomaly_score}, is_fraud={is_fraud}")
            return response

        except Exception as e:
            if self.db:
                try:
                    self.db.rollback()
                except Exception:
                    pass
            logger.exception(f"Error predicting single transaction: {e}")
            raise PredictionError(f"Prediction failed: {str(e)}") from e

    def predict_batch(
        self,
        transactions: List[TransactionRequest],
        user_id: Optional[str] = None,
        ip_address: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Processes, scores, and atomically persists a batch of ERP transactions.
        """
        try:
            logger.info(f"Processing batch of {len(transactions)} transactions.")
            results = []
            for txn in transactions:
                # We disable separate prediction audit logs to keep log file sizes clean.
                results.append(
                    self.predict_single(
                        txn,
                        user_id=user_id,
                        ip_address=ip_address,
                        disable_audit=True,
                    )
                )

            flagged = [r for r in results if r.is_fraud]
            total = len(results)
            flagged_count = len(flagged)
            flag_rate_pct = (flagged_count / total * 100) if total > 0 else 0.0

            # Create batch audit log
            if self.audit_log_repo:
                audit_data = {
                    "user_id": user_id,
                    "action": "BATCH_PREDICT",
                    "resource": "/v1/predict/batch",
                    "details": {
                        "total": total,
                        "flagged": flagged_count,
                        "flag_rate": f"{flag_rate_pct:.1f}%",
                    },
                    "ip_address": ip_address,
                }
                self.audit_log_repo.create(audit_data)

            if self.db:
                self.db.commit()

            return {
                "total": total,
                "flagged": flagged_count,
                "flag_rate": f"{flag_rate_pct:.1f}%",
                "predictions": results
            }
        except Exception as e:
            if self.db:
                try:
                    self.db.rollback()
                except Exception:
                    pass
            logger.exception(f"Error predicting batch: {e}")
            raise BatchPredictionError(f"Batch prediction failed: {str(e)}") from e
