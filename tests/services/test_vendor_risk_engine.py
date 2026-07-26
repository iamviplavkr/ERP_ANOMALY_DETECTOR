"""
tests/services/test_vendor_risk_engine.py
─────────────────────────────────────────────────────────────────
Tests for vendor-signal rules in the Risk Engine.
Verifies Rules 6-9: blacklist, watchlist, low reputation, and
high historical alert frequency all correctly affect risk level.
"""

import pytest
from backend.services.risk_engine import evaluate_risk

# Base transaction that does not trigger any ML-based rules on its own
_LOW_TX = {"vendor_id": "V001", "department": "Finance", "transaction_amount": 100.0}
_LOW_SCORE = 0.1
_NO_FACTORS: list = []


class TestVendorRiskRules:
    """Each test verifies a single vendor-signal rule in isolation."""

    # ── Rule 6: Blacklisted Vendor (CRITICAL) ────────────────────────────────

    def test_blacklisted_vendor_triggers_critical(self):
        vendor = {"is_blacklisted": True, "is_watchlist": False, "reputation_score": 90.0, "historical_alerts_count": 0}
        level, rules, mitigation = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS, vendor_risk=vendor)
        assert level == "CRITICAL"
        assert any("Blacklisted" in r for r in rules)
        assert "blacklisted" in mitigation.lower()

    def test_blacklisted_vendor_overrides_lower_severity(self):
        """Even with medium-level ML score, blacklisted → CRITICAL."""
        vendor = {"is_blacklisted": True, "is_watchlist": False, "reputation_score": 90.0, "historical_alerts_count": 0}
        level, rules, mitigation = evaluate_risk(_LOW_TX, 0.55, _NO_FACTORS, vendor_risk=vendor)
        assert level == "CRITICAL"

    # ── Rule 7: Watchlist Vendor (HIGH) ─────────────────────────────────────

    def test_watchlist_vendor_triggers_high(self):
        vendor = {"is_blacklisted": False, "is_watchlist": True, "reputation_score": 90.0, "historical_alerts_count": 0}
        level, rules, mitigation = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS, vendor_risk=vendor)
        assert level == "HIGH"
        assert any("Watchlist" in r for r in rules)

    def test_blacklisted_and_watchlist_still_critical(self):
        """Blacklisted takes precedence over watchlist → CRITICAL."""
        vendor = {"is_blacklisted": True, "is_watchlist": True, "reputation_score": 90.0, "historical_alerts_count": 0}
        level, rules, _ = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS, vendor_risk=vendor)
        assert level == "CRITICAL"

    # ── Rule 8: Low Vendor Reputation (HIGH / MEDIUM) ───────────────────────

    def test_very_low_reputation_triggers_high(self):
        vendor = {"is_blacklisted": False, "is_watchlist": False, "reputation_score": 20.0, "historical_alerts_count": 0}
        level, rules, _ = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS, vendor_risk=vendor)
        assert level == "HIGH"
        assert any("Low Vendor Reputation (High)" in r for r in rules)

    def test_low_reputation_triggers_medium(self):
        vendor = {"is_blacklisted": False, "is_watchlist": False, "reputation_score": 40.0, "historical_alerts_count": 0}
        level, rules, _ = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS, vendor_risk=vendor)
        assert level == "MEDIUM"
        assert any("Low Vendor Reputation (Medium)" in r for r in rules)

    def test_good_reputation_does_not_trigger(self):
        vendor = {"is_blacklisted": False, "is_watchlist": False, "reputation_score": 85.0, "historical_alerts_count": 0}
        level, rules, _ = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS, vendor_risk=vendor)
        assert level == "LOW"
        assert not any("Reputation" in r for r in rules)

    # ── Rule 9: High Historical Alerts (MEDIUM) ──────────────────────────────

    def test_high_historical_alerts_triggers_medium(self):
        vendor = {"is_blacklisted": False, "is_watchlist": False, "reputation_score": 90.0, "historical_alerts_count": 5}
        level, rules, _ = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS, vendor_risk=vendor)
        assert level == "MEDIUM"
        assert any("Historical Alerts" in r for r in rules)

    def test_low_historical_alerts_does_not_trigger(self):
        vendor = {"is_blacklisted": False, "is_watchlist": False, "reputation_score": 90.0, "historical_alerts_count": 2}
        level, rules, _ = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS, vendor_risk=vendor)
        assert level == "LOW"
        assert not any("Historical Alerts" in r for r in rules)

    # ── vendor_risk=None preserves backward compatibility ────────────────────

    def test_no_vendor_risk_does_not_affect_low_score(self):
        level, rules, _ = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS, vendor_risk=None)
        assert level == "LOW"
        assert rules == []

    def test_no_vendor_risk_kwarg_still_works(self):
        """Calling without vendor_risk keyword must be backward-compatible."""
        level, rules, _ = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS)
        assert level == "LOW"

    # ── Combined rule interaction ─────────────────────────────────────────────

    def test_watchlist_plus_historical_alerts_stays_high(self):
        vendor = {"is_blacklisted": False, "is_watchlist": True, "reputation_score": 90.0, "historical_alerts_count": 5}
        level, rules, _ = evaluate_risk(_LOW_TX, _LOW_SCORE, _NO_FACTORS, vendor_risk=vendor)
        assert level == "HIGH"  # Watchlist (HIGH) beats Historical Alerts (MEDIUM)
        assert any("Watchlist" in r for r in rules)
        assert any("Historical Alerts" in r for r in rules)
