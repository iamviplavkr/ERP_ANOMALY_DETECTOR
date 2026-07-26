"""
backend/services/analytics_service.py
─────────────────────────────────────────────────────────────────
Service layer implementing cached BI and Executive Analytics queries.

Caching strategy
────────────────
When ANALYTICS_CACHE_TTL_SECONDS > 0 results are kept in a
module-level dict for that many seconds before the next DB hit.
Set ANALYTICS_CACHE_TTL_SECONDS=0 (the default) to disable the
cache entirely — every request then hits PostgreSQL directly and
newly-inserted rows are always visible.

Call ``AnalyticsService.invalidate_cache()`` (e.g. from the predict
endpoint) to flush all cached entries immediately.
"""

from datetime import datetime, timedelta, timezone
import hashlib
import json
from typing import Any, Dict, List, Tuple

from backend.core.config import settings
from backend.repositories.analytics_repository import AnalyticsRepositoryInterface


# ── Module-level TTL cache ────────────────────────────────────────────────────
# Keyed by (method_name, hashed_args) → (result, expiry_datetime)
_ANALYTICS_CACHE: Dict[str, Tuple[Any, datetime]] = {}


def invalidate_cache() -> None:
    """Flush every cached analytics result immediately.

    Call this after any write that should be reflected in the BI
    dashboard without waiting for TTL expiry (e.g. after a successful
    /v1/predict call).
    """
    _ANALYTICS_CACHE.clear()


class AnalyticsService:
    def __init__(self, analytics_repo: AnalyticsRepositoryInterface) -> None:
        self.analytics_repo = analytics_repo

    @staticmethod
    def invalidate_cache() -> None:
        """Class-level alias for the module-level ``invalidate_cache()``."""
        invalidate_cache()

    def _get_cache_key(self, func_name: str, *args, **kwargs) -> str:
        # Create a unique key by hashing parameters
        serialized = json.dumps(
            {"args": [str(a) for a in args], "kwargs": {k: str(v) for k, v in kwargs.items()}},
            sort_keys=True
        )
        hasher = hashlib.sha256()
        hasher.update(serialized.encode("utf-8"))
        return f"{func_name}:{hasher.hexdigest()}"

    def _get_cached_or_execute(self, func_name: str, repo_func: Any, *args, **kwargs) -> Any:
        ttl = settings.ANALYTICS_CACHE_TTL_SECONDS
        if ttl <= 0:
            # Cache disabled — always query the database
            return repo_func(*args, **kwargs)

        now = datetime.now(timezone.utc)
        cache_key = self._get_cache_key(func_name, *args, **kwargs)

        if cache_key in _ANALYTICS_CACHE:
            val, expiry = _ANALYTICS_CACHE[cache_key]
            if now < expiry:
                return val
            else:
                del _ANALYTICS_CACHE[cache_key]

        # Execute fresh query then store with expiry
        result = repo_func(*args, **kwargs)
        expiry = now + timedelta(seconds=ttl)
        _ANALYTICS_CACHE[cache_key] = (result, expiry)
        return result

    def get_kpis(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        return self._get_cached_or_execute(
            "get_kpis", self.analytics_repo.get_kpis, start_date, end_date
        )

    def get_fraud_trends(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return self._get_cached_or_execute(
            "get_fraud_trends", self.analytics_repo.get_fraud_trends, start_date, end_date
        )

    def get_risk_distribution(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return self._get_cached_or_execute(
            "get_risk_distribution", self.analytics_repo.get_risk_distribution, start_date, end_date
        )

    def get_department_metrics(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return self._get_cached_or_execute(
            "get_department_metrics", self.analytics_repo.get_department_metrics, start_date, end_date
        )

    def get_vendor_rankings(
        self,
        start_date: datetime,
        end_date: datetime,
        limit: int = 10,
        offset: int = 0,
        sort_by: str = "reputation_score",
        sort_order: str = "asc",
    ) -> Dict[str, Any]:
        return self._get_cached_or_execute(
            "get_vendor_rankings",
            self.analytics_repo.get_vendor_rankings,
            start_date,
            end_date,
            limit=limit,
            offset=offset,
            sort_by=sort_by,
            sort_order=sort_order,
        )

    def get_alert_lifecycle(self, start_date: datetime, end_date: datetime) -> List[Dict[str, Any]]:
        return self._get_cached_or_execute(
            "get_alert_lifecycle", self.analytics_repo.get_alert_lifecycle, start_date, end_date
        )

    def get_model_performance(self, start_date: datetime, end_date: datetime) -> Dict[str, Any]:
        return self._get_cached_or_execute(
            "get_model_performance", self.analytics_repo.get_model_performance, start_date, end_date
        )

    def get_predictions_export(
        self, start_date: datetime, end_date: datetime, sort_by: str = "created_at", sort_order: str = "desc"
    ) -> List[Dict[str, Any]]:
        # Since CSV exports are large and need to be real-time/audited, we can still cache if needed
        # or skip caching. Let's apply caching to export as well since the instruction requests:
        # "Add configurable response caching for analytics queries."
        return self._get_cached_or_execute(
            "get_predictions_export",
            self.analytics_repo.get_predictions_export,
            start_date,
            end_date,
            sort_by=sort_by,
            sort_order=sort_order,
        )
