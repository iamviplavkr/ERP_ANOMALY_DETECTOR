"""
tests/services/test_risk_engine.py
─────────────────────────────────────────────────────────────────
Tests for the configurable rule-based risk engine.
"""

import pytest
from backend.services.risk_engine import evaluate_risk


class TestEvaluateRisk:
    """Test each rule in the risk engine independently."""

    def test_low_score_no_rules_triggered(self):
        level, rules, mitigation = evaluate_risk(
            transaction_data={"vendor_id": "V001", "department": "Finance", "transaction_amount": 100.0},
            anomaly_score=0.1,
            top_risk_factors=[],
        )
        assert level == "LOW"
        assert rules == []

    def test_fraud_threshold_triggers_medium(self):
        level, rules, mitigation = evaluate_risk(
            transaction_data={"vendor_id": "V001", "department": "Finance", "transaction_amount": 100.0},
            anomaly_score=0.55,  # above FRAUD_THRESHOLD=0.5, below HIGH=0.8
            top_risk_factors=[],
        )
        assert level == "MEDIUM"
        assert len(rules) == 1
        assert "ML Anomaly Score" in rules[0]

    def test_high_anomaly_triggers_critical(self):
        level, rules, mitigation = evaluate_risk(
            transaction_data={"vendor_id": "V001", "department": "Finance", "transaction_amount": 100.0},
            anomaly_score=0.9,  # above RISK_RULE_ANOMALY_HIGH_THRESHOLD=0.8
            top_risk_factors=[],
        )
        assert level == "CRITICAL"
        assert any("High Anomaly Score" in r for r in rules)

    def test_large_amount_triggers_high(self):
        level, rules, mitigation = evaluate_risk(
            transaction_data={"vendor_id": "V001", "department": "Finance", "transaction_amount": 75000.0},
            anomaly_score=0.4,  # above LARGE_AMOUNT_ANOMALY_THRESHOLD=0.3
            top_risk_factors=[],
        )
        assert level == "HIGH"
        assert any("Large Transaction Amount" in r for r in rules)

    def test_procurement_triggers_high(self):
        level, rules, mitigation = evaluate_risk(
            transaction_data={"vendor_id": "V001", "department": "Procurement", "transaction_amount": 500.0},
            anomaly_score=0.6,
            top_risk_factors=[],
        )
        assert level == "HIGH"
        assert any("Procurement" in r for r in rules)

    def test_suspicious_vendor_triggers_medium(self):
        level, rules, mitigation = evaluate_risk(
            transaction_data={"vendor_id": "UNKNOWN", "department": "Finance", "transaction_amount": 100.0},
            anomaly_score=0.55,
            top_risk_factors=[],
        )
        assert level == "MEDIUM"
        assert any("Suspicious Vendor" in r for r in rules)

    def test_susp_prefix_vendor_triggers_medium(self):
        level, rules, mitigation = evaluate_risk(
            transaction_data={"vendor_id": "SUSP_001", "department": "Finance", "transaction_amount": 100.0},
            anomaly_score=0.5,
            top_risk_factors=[],
        )
        assert level == "MEDIUM"
        assert any("Suspicious Vendor" in r for r in rules)

    def test_amount_risk_factor_triggers_medium(self):
        level, rules, mitigation = evaluate_risk(
            transaction_data={"vendor_id": "V001", "department": "Finance", "transaction_amount": 100.0},
            anomaly_score=0.6,
            top_risk_factors=[{"feature": "Amount", "influence": 0.7}],
        )
        assert level == "MEDIUM"
        assert any("High Risk Factor" in r for r in rules)

    def test_critical_overrides_high(self):
        """Critical anomaly score should override large-amount high risk."""
        level, rules, mitigation = evaluate_risk(
            transaction_data={"vendor_id": "V001", "department": "Finance", "transaction_amount": 75000.0},
            anomaly_score=0.95,
            top_risk_factors=[],
        )
        assert level == "CRITICAL"

    def test_mitigation_not_empty(self):
        _, _, mitigation = evaluate_risk(
            transaction_data={"vendor_id": "V001", "department": "Finance", "transaction_amount": 100.0},
            anomaly_score=0.9,
            top_risk_factors=[],
        )
        assert len(mitigation) > 0
