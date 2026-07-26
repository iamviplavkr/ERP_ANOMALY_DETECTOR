"""
backend/repositories/analytics_repository.py
─────────────────────────────────────────────────────────────────
Analytics repository interface, Postgres implementation, and InMemory stub.
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional
import uuid

from sqlalchemy import func, case
from sqlalchemy.orm import Session

from backend.core.logging import get_logger
from backend.models.transaction import TransactionModel
from backend.models.prediction import PredictionModel
from backend.models.alert import AlertModel
from backend.models.vendor import VendorRiskModel

logger = get_logger(__name__)


class AnalyticsRepositoryInterface(ABC):
    @abstractmethod
    def get_kpis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_fraud_trends(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_risk_distribution(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_department_metrics(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_vendor_rankings(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 10,
        offset: int = 0,
        sort_by: str = "reputation_score",
        sort_order: str = "asc",
    ) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_alert_lifecycle(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        pass

    @abstractmethod
    def get_model_performance(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_predictions_export(
        self, start_date: datetime, end_date: datetime, sort_by: str = "created_at", sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        pass


class PostgresAnalyticsRepository(AnalyticsRepositoryInterface):
    def __init__(self, db: Session) -> None:
        self.db = db

    def get_kpis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        # Total transactions & amount
        tx_stats = self.db.query(
            func.count(TransactionModel.id).label("total_count"),
            func.sum(TransactionModel.transaction_amount).label("total_amount"),
        ).filter(
            TransactionModel.created_at >= start_date,
            TransactionModel.created_at <= end_date,
        ).first()

        total_tx = tx_stats.total_count if tx_stats and tx_stats.total_count is not None else 0
        total_amt = float(tx_stats.total_amount) if tx_stats and tx_stats.total_amount is not None else 0.0

        # Flagged anomalies count
        flagged_count = self.db.query(func.count(PredictionModel.id)).filter(
            PredictionModel.created_at >= start_date,
            PredictionModel.created_at <= end_date,
            PredictionModel.is_fraud == True,
        ).scalar() or 0

        # Open alerts count
        open_alerts = self.db.query(func.count(AlertModel.id)).filter(
            AlertModel.created_at >= start_date,
            AlertModel.created_at <= end_date,
            AlertModel.status == "OPEN",
        ).scalar() or 0

        # Total unique active vendors
        total_vendors = self.db.query(func.count(func.distinct(TransactionModel.vendor_id))).filter(
            TransactionModel.created_at >= start_date,
            TransactionModel.created_at <= end_date,
        ).scalar() or 0

        # Average anomaly score
        avg_score = self.db.query(func.avg(PredictionModel.anomaly_score)).filter(
            PredictionModel.created_at >= start_date,
            PredictionModel.created_at <= end_date,
        ).scalar() or 0.0

        logger.debug(
            "[ANALYTICS-KPI] range=[%s, %s] total_tx=%d flagged=%d open_alerts=%d vendors=%d avg_score=%.4f",
            start_date.isoformat(), end_date.isoformat(),
            total_tx, flagged_count, open_alerts, total_vendors, float(avg_score),
        )

        return {
            "total_transactions": total_tx,
            "total_amount": total_amt,
            "flagged_anomalies": flagged_count,
            "open_alerts": open_alerts,
            "total_vendors": total_vendors,
            "average_anomaly_score": float(avg_score),
            "anomaly_rate": (flagged_count / total_tx) if total_tx > 0 else 0.0,
        }

    def get_fraud_trends(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        # Use func.date for engine cross-compatibility
        results = self.db.query(
            func.date(PredictionModel.created_at).label("date_label"),
            func.count(PredictionModel.id).label("count"),
            func.avg(PredictionModel.anomaly_score).label("avg_score"),
            func.sum(case((PredictionModel.is_fraud == True, 1), else_=0)).label("fraud_count"),
        ).filter(
            PredictionModel.created_at >= start_date,
            PredictionModel.created_at <= end_date,
        ).group_by(
            func.date(PredictionModel.created_at)
        ).order_by(
            func.date(PredictionModel.created_at).asc()
        ).all()

        return [
            {
                "date": str(r.date_label),
                "count": r.count,
                "average_anomaly_score": float(r.avg_score or 0.0),
                "flagged_count": int(r.fraud_count or 0),
            }
            for r in results
        ]

    def get_risk_distribution(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        results = self.db.query(
            PredictionModel.risk_level,
            func.count(PredictionModel.id).label("count"),
            func.sum(TransactionModel.transaction_amount).label("total_amount"),
        ).join(
            TransactionModel, PredictionModel.transaction_id == TransactionModel.id
        ).filter(
            PredictionModel.created_at >= start_date,
            PredictionModel.created_at <= end_date,
        ).group_by(
            PredictionModel.risk_level
        ).all()

        return [
            {
                "risk_level": r.risk_level or "UNKNOWN",
                "count": r.count,
                "total_amount": float(r.total_amount or 0.0),
            }
            for r in results
        ]

    def get_department_metrics(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        results = self.db.query(
            TransactionModel.department,
            func.count(TransactionModel.id).label("count"),
            func.sum(TransactionModel.transaction_amount).label("total_amount"),
            func.avg(PredictionModel.anomaly_score).label("avg_score"),
            func.sum(case((PredictionModel.is_fraud == True, 1), else_=0)).label("fraud_count"),
        ).join(
            PredictionModel, PredictionModel.transaction_id == TransactionModel.id
        ).filter(
            TransactionModel.created_at >= start_date,
            TransactionModel.created_at <= end_date,
        ).group_by(
            TransactionModel.department
        ).all()

        return [
            {
                "department": r.department or "Unknown",
                "count": r.count,
                "total_amount": float(r.total_amount or 0.0),
                "average_anomaly_score": float(r.avg_score or 0.0),
                "flagged_count": int(r.fraud_count or 0),
            }
            for r in results
        ]

    def get_vendor_rankings(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 10,
        offset: int = 0,
        sort_by: str = "reputation_score",
        sort_order: str = "asc",
    ) -> Dict[str, Any]:
        # Filter rankings by vendors active in range or overall
        q = self.db.query(VendorRiskModel)

        # Apply sorting dynamically
        col = getattr(VendorRiskModel, sort_by, None)
        if col is not None:
            if sort_order.lower() == "desc":
                q = q.order_by(col.desc())
            else:
                q = q.order_by(col.asc())
        else:
            q = q.order_by(VendorRiskModel.reputation_score.asc())

        total = q.count()
        results = q.offset(offset).limit(limit).all()

        vendors_list = [
            {
                "id": str(v.id),
                "vendor_id": v.vendor_id,
                "name": v.name,
                "reputation_score": v.reputation_score,
                "is_blacklisted": v.is_blacklisted,
                "is_watchlist": v.is_watchlist,
                "historical_alerts_count": v.historical_alerts_count,
                "total_transactions_count": v.total_transactions_count,
                "historical_fraud_rate": v.historical_fraud_rate,
                "last_transaction_at": v.last_transaction_at.isoformat() if v.last_transaction_at else None,
                "last_alert_at": v.last_alert_at.isoformat() if v.last_alert_at else None,
            }
            for v in results
        ]

        return {"total": total, "vendors": vendors_list}

    def get_alert_lifecycle(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        results = self.db.query(
            AlertModel.status,
            func.count(AlertModel.id).label("count"),
        ).filter(
            AlertModel.created_at >= start_date,
            AlertModel.created_at <= end_date,
        ).group_by(
            AlertModel.status
        ).all()

        return [{"status": r.status, "count": r.count} for r in results]

    def get_model_performance(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        # Simple model summary analytics
        stats = self.db.query(
            func.count(PredictionModel.id).label("total"),
            func.avg(PredictionModel.anomaly_score).label("avg_score"),
            func.max(PredictionModel.anomaly_score).label("max_score"),
        ).filter(
            PredictionModel.created_at >= start_date,
            PredictionModel.created_at <= end_date,
        ).first()

        total = stats.total if stats and stats.total is not None else 0
        avg_score = float(stats.avg_score) if stats and stats.avg_score is not None else 0.0
        max_score = float(stats.max_score) if stats and stats.max_score is not None else 0.0

        # Score distributions
        high_score_count = self.db.query(func.count(PredictionModel.id)).filter(
            PredictionModel.created_at >= start_date,
            PredictionModel.created_at <= end_date,
            PredictionModel.anomaly_score >= 0.5,
        ).scalar() or 0

        low_score_count = total - high_score_count

        return {
            "total_predictions": total,
            "average_score": avg_score,
            "max_score": max_score,
            "high_confidence_anomalies": high_score_count,
            "low_confidence_normal": low_score_count,
        }

    def get_predictions_export(
        self, start_date: datetime, end_date: datetime, sort_by: str = "created_at", sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        q = self.db.query(
            PredictionModel.id.label("prediction_id"),
            PredictionModel.anomaly_score,
            PredictionModel.is_fraud,
            PredictionModel.risk_level,
            PredictionModel.alert_message,
            PredictionModel.model_version,
            PredictionModel.created_at.label("pred_created_at"),
            TransactionModel.id.label("transaction_id"),
            TransactionModel.vendor_id,
            TransactionModel.department,
            TransactionModel.approved_by,
            TransactionModel.transaction_amount,
        ).join(
            TransactionModel, PredictionModel.transaction_id == TransactionModel.id
        ).filter(
            PredictionModel.created_at >= start_date,
            PredictionModel.created_at <= end_date,
        )

        col = getattr(PredictionModel, sort_by, None)
        if col is None:
            col = getattr(TransactionModel, sort_by, None)
        if col is not None:
            if sort_order.lower() == "desc":
                q = q.order_by(col.desc())
            else:
                q = q.order_by(col.asc())
        else:
            q = q.order_by(PredictionModel.created_at.desc())

        results = q.all()

        return [
            {
                "prediction_id": str(r.prediction_id),
                "transaction_id": str(r.transaction_id),
                "vendor_id": r.vendor_id,
                "department": r.department,
                "approved_by": r.approved_by,
                "transaction_amount": float(r.transaction_amount or 0.0),
                "anomaly_score": float(r.anomaly_score or 0.0),
                "is_fraud": r.is_fraud,
                "risk_level": r.risk_level,
                "alert_message": r.alert_message,
                "model_version": r.model_version,
                "created_at": r.pred_created_at.isoformat() if r.pred_created_at else None,
            }
            for r in results
        ]


class InMemoryAnalyticsRepository(AnalyticsRepositoryInterface):
    def __init__(self, transaction_repo: Any, prediction_repo: Any, alert_repo: Any, vendor_repo: Any) -> None:
        self.transaction_repo = transaction_repo
        self.prediction_repo = prediction_repo
        self.alert_repo = alert_repo
        self.vendor_repo = vendor_repo

    def _parse_date(self, val: Any) -> datetime:
        from datetime import timezone
        dt = None
        if isinstance(val, datetime):
            dt = val
        elif isinstance(val, str):
            try:
                dt = datetime.fromisoformat(val.replace("Z", "+00:00"))
            except ValueError:
                pass

        if dt is None:
            dt = datetime.now(timezone.utc)

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt


    def get_kpis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        txs = self.transaction_repo.list_all() if hasattr(self.transaction_repo, "list_all") else list(self.transaction_repo._transactions.values())
        preds = list(self.prediction_repo._predictions.values())
        alerts = list(self.alert_repo._alerts.values())

        # Filter transactions
        filtered_txs = [
            t for t in txs
            if start_date <= self._parse_date(t.get("created_at")) <= end_date
        ]
        total_tx = len(filtered_txs)
        total_amt = sum(float(t.get("transaction_amount") or 0.0) for t in filtered_txs)

        # Filter predictions
        filtered_preds = [
            p for p in preds
            if start_date <= self._parse_date(p.get("created_at")) <= end_date
        ]
        flagged_count = sum(1 for p in filtered_preds if p.get("is_fraud"))
        avg_score = sum(float(p.get("anomaly_score") or 0.0) for p in filtered_preds) / len(filtered_preds) if filtered_preds else 0.0

        # Filter open alerts
        filtered_alerts = [
            a for a in alerts
            if start_date <= self._parse_date(a.get("created_at")) <= end_date and a.get("status") == "OPEN"
        ]
        open_alerts = len(filtered_alerts)

        # Unique active vendors count
        active_vendors = len(set(t.get("vendor_id") for t in filtered_txs if t.get("vendor_id")))

        return {
            "total_transactions": total_tx,
            "total_amount": total_amt,
            "flagged_anomalies": flagged_count,
            "open_alerts": open_alerts,
            "total_vendors": active_vendors,
            "average_anomaly_score": avg_score,
            "anomaly_rate": (flagged_count / total_tx) if total_tx > 0 else 0.0,
        }

    def get_fraud_trends(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        preds = list(self.prediction_repo._predictions.values())
        filtered_preds = [
            p for p in preds
            if start_date <= self._parse_date(p.get("created_at")) <= end_date
        ]

        # Group by date string
        by_day: Dict[str, List[Dict[str, Any]]] = {}
        for p in filtered_preds:
            d_str = self._parse_date(p.get("created_at")).date().isoformat()
            by_day.setdefault(d_str, []).append(p)

        results = []
        for d_str, day_preds in sorted(by_day.items()):
            results.append({
                "date": d_str,
                "count": len(day_preds),
                "average_anomaly_score": sum(float(p.get("anomaly_score") or 0.0) for p in day_preds) / len(day_preds),
                "flagged_count": sum(1 for p in day_preds if p.get("is_fraud")),
            })
        return results

    def get_risk_distribution(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        preds = list(self.prediction_repo._predictions.values())
        txs_dict = {t["id"]: t for t in (self.transaction_repo.list_all() if hasattr(self.transaction_repo, "list_all") else list(self.transaction_repo._transactions.values()))}

        filtered = [
            p for p in preds
            if start_date <= self._parse_date(p.get("created_at")) <= end_date
        ]

        by_risk: Dict[str, List[Dict[str, Any]]] = {}
        for p in filtered:
            by_risk.setdefault(p.get("risk_level") or "UNKNOWN", []).append(p)

        results = []
        for risk, r_preds in by_risk.items():
            tot_amt = sum(float(txs_dict.get(p["transaction_id"], {}).get("transaction_amount") or 0.0) for p in r_preds if p["transaction_id"] in txs_dict)
            results.append({
                "risk_level": risk,
                "count": len(r_preds),
                "total_amount": tot_amt,
            })
        return results

    def get_department_metrics(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        preds_dict = {p["transaction_id"]: p for p in list(self.prediction_repo._predictions.values())}
        txs = self.transaction_repo.list_all() if hasattr(self.transaction_repo, "list_all") else list(self.transaction_repo._transactions.values())

        filtered_txs = [
            t for t in txs
            if start_date <= self._parse_date(t.get("created_at")) <= end_date
        ]

        by_dept: Dict[str, List[Dict[str, Any]]] = {}
        for t in filtered_txs:
            by_dept.setdefault(t.get("department") or "Unknown", []).append(t)

        results = []
        for dept, dept_txs in by_dept.items():
            count = len(dept_txs)
            tot_amt = sum(float(t.get("transaction_amount") or 0.0) for t in dept_txs)
            
            # Map predictions
            scores = []
            flagged = 0
            for t in dept_txs:
                pred = preds_dict.get(t["id"])
                if pred:
                    scores.append(float(pred.get("anomaly_score") or 0.0))
                    if pred.get("is_fraud"):
                        flagged += 1

            results.append({
                "department": dept,
                "count": count,
                "total_amount": tot_amt,
                "average_anomaly_score": sum(scores) / len(scores) if scores else 0.0,
                "flagged_count": flagged,
            })
        return results

    def get_vendor_rankings(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 10,
        offset: int = 0,
        sort_by: str = "reputation_score",
        sort_order: str = "asc",
    ) -> Dict[str, Any]:
        vendors = list(self.vendor_repo._vendors.values())

        # Sort Python-side
        reverse = (sort_order.lower() == "desc")
        vendors.sort(key=lambda x: x.get(sort_by) if x.get(sort_by) is not None else 0.0, reverse=reverse)

        total = len(vendors)
        paginated = vendors[offset : offset + limit]

        vendors_list = [
            {
                "id": v.get("id"),
                "vendor_id": v.get("vendor_id"),
                "name": v.get("name"),
                "reputation_score": v.get("reputation_score"),
                "is_blacklisted": v.get("is_blacklisted"),
                "is_watchlist": v.get("is_watchlist"),
                "historical_alerts_count": v.get("historical_alerts_count"),
                "total_transactions_count": v.get("total_transactions_count"),
                "historical_fraud_rate": v.get("historical_fraud_rate"),
                "last_transaction_at": v.get("last_transaction_at"),
                "last_alert_at": v.get("last_alert_at"),
            }
            for v in paginated
        ]

        return {"total": total, "vendors": vendors_list}

    def get_alert_lifecycle(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        alerts = list(self.alert_repo._alerts.values())
        filtered = [
            a for a in alerts
            if start_date <= self._parse_date(a.get("created_at")) <= end_date
        ]

        by_status: Dict[str, int] = {}
        for a in filtered:
            status = a.get("status") or "OPEN"
            by_status[status] = by_status.get(status, 0) + 1

        return [{"status": status, "count": count} for status, count in by_status.items()]

    def get_model_performance(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        preds = list(self.prediction_repo._predictions.values())
        filtered = [
            p for p in preds
            if start_date <= self._parse_date(p.get("created_at")) <= end_date
        ]

        total = len(filtered)
        if total == 0:
            return {
                "total_predictions": 0,
                "average_score": 0.0,
                "max_score": 0.0,
                "high_confidence_anomalies": 0,
                "low_confidence_normal": 0,
            }

        scores = [float(p.get("anomaly_score") or 0.0) for p in filtered]
        avg_score = sum(scores) / total
        max_score = max(scores)
        high_confidence = sum(1 for s in scores if s >= 0.5)

        return {
            "total_predictions": total,
            "average_score": avg_score,
            "max_score": max_score,
            "high_confidence_anomalies": high_confidence,
            "low_confidence_normal": total - high_confidence,
        }

    def get_predictions_export(
        self, start_date: datetime, end_date: datetime, sort_by: str = "created_at", sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        preds = list(self.prediction_repo._predictions.values())
        txs_dict = {t["id"]: t for t in (self.transaction_repo.list_all() if hasattr(self.transaction_repo, "list_all") else list(self.transaction_repo._transactions.values()))}

        filtered = [
            p for p in preds
            if start_date <= self._parse_date(p.get("created_at")) <= end_date
        ]

        export_list = []
        for p in filtered:
            tx = txs_dict.get(p["transaction_id"], {})
            export_list.append({
                "prediction_id": p["id"],
                "transaction_id": p["transaction_id"],
                "vendor_id": tx.get("vendor_id"),
                "department": tx.get("department"),
                "approved_by": tx.get("approved_by"),
                "transaction_amount": float(tx.get("transaction_amount") or 0.0),
                "anomaly_score": float(p.get("anomaly_score") or 0.0),
                "is_fraud": p.get("is_fraud"),
                "risk_level": p.get("risk_level"),
                "alert_message": p.get("alert_message"),
                "model_version": p.get("model_version"),
                "created_at": p.get("created_at").isoformat() if isinstance(p.get("created_at"), datetime) else p.get("created_at"),
            })

        reverse = (sort_order.lower() == "desc")
        export_list.sort(key=lambda x: x.get(sort_by) if x.get(sort_by) is not None else "", reverse=reverse)
        return export_list
