"""
backend/services/vendor_service.py
─────────────────────────────────────────────────────────────────
Service layer implementation for managing vendor profiles and external risk.
"""

from typing import Dict, Any, List, Optional
from backend.repositories.vendor_repository import VendorRepositoryInterface


class VendorService:
    def __init__(self, vendor_repo: VendorRepositoryInterface) -> None:
        self.vendor_repo = vendor_repo

    def get_by_vendor_id(self, vendor_id: str) -> Optional[Dict[str, Any]]:
        return self.vendor_repo.get_by_vendor_id(vendor_id)

    def create_vendor(self, vendor_data: Dict[str, Any]) -> Dict[str, Any]:
        return self.vendor_repo.create(vendor_data)

    def update_vendor(self, vendor_id: str, updates: Dict[str, Any]) -> Dict[str, Any]:
        return self.vendor_repo.update(vendor_id, updates)

    def delete_vendor(self, vendor_id: str) -> bool:
        return self.vendor_repo.delete(vendor_id)

    def list_vendors(
        self,
        is_blacklisted: Optional[bool] = None,
        is_watchlist: Optional[bool] = None
    ) -> List[Dict[str, Any]]:
        return self.vendor_repo.list_all(
            is_blacklisted=is_blacklisted,
            is_watchlist=is_watchlist
        )

    def get_or_create_default(self, vendor_id: str) -> Dict[str, Any]:
        """
        Retrieves vendor profile by ID. If not found, dynamically initializes
        a default vendor profile in the database.
        """
        profile = self.vendor_repo.get_by_vendor_id(vendor_id)
        if not profile:
            default_data = {
                "vendor_id": vendor_id,
                "name": f"New Vendor ({vendor_id})",
                "reputation_score": 90.0,
                "is_blacklisted": False,
                "is_watchlist": False,
                "historical_alerts_count": 0,
                "total_transactions_count": 0,
                "historical_fraud_rate": 0.0,
                "last_transaction_at": None,
                "last_alert_at": None,
            }
            profile = self.vendor_repo.create(default_data)
        return profile
