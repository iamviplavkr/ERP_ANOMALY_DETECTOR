"""
backend/services/risk_engine.py
─────────────────────────────────────────────────────────────────
Configurable rule-based risk evaluation engine on top of ML predictions.
Evaluates anomaly score, amount, department, vendor, and factors.
"""

from typing import Dict, Any, List, Tuple, Optional
from backend.core.config import settings


def evaluate_risk(
    transaction_data: Dict[str, Any],
    anomaly_score: float,
    top_risk_factors: List[Dict[str, Any]],
    vendor_risk: Optional[Dict[str, Any]] = None
) -> Tuple[str, List[str], str]:
    """
    Evaluates rule-based risks and returns:
    (max_risk_level, rules_triggered, recommended_mitigation)

    Risk severity levels: LOW, MEDIUM, HIGH, CRITICAL.
    """
    triggered_rules = []
    highest_severity = "LOW"
    mitigation_actions = []

    amount = float(transaction_data.get("transaction_amount") or 0.0)
    dept = transaction_data.get("department") or ""
    vendor_id = transaction_data.get("vendor_id") or ""

    # Rule 1: High Anomaly Score Rule (CRITICAL)
    if anomaly_score >= settings.RISK_RULE_ANOMALY_HIGH_THRESHOLD:
        triggered_rules.append("High Anomaly Score (Critical)")
        highest_severity = "CRITICAL"
        mitigation_actions.append(
            "Immediate freeze of transaction and suspend user/vendor account. Route for urgent review."
        )

    # Rule 2: Large Transaction Amount Rule (HIGH)
    if (
        amount >= settings.RISK_RULE_LARGE_AMOUNT_THRESHOLD
        and anomaly_score >= settings.RISK_RULE_LARGE_AMOUNT_ANOMALY_THRESHOLD
    ):
        triggered_rules.append("Large Transaction Amount (High)")
        if highest_severity != "CRITICAL":
            highest_severity = "HIGH"
        mitigation_actions.append(
            "Request manual verification of invoices and executive sign-off."
        )

    # Rule 3: Department Risk Rule (HIGH)
    if (
        dept.lower() == "procurement"
        and anomaly_score >= settings.RISK_RULE_PROCUREMENT_ANOMALY_THRESHOLD
    ):
        triggered_rules.append("Procurement Anomaly (High)")
        if highest_severity not in ["CRITICAL", "HIGH"]:
            highest_severity = "HIGH"
        mitigation_actions.append(
            "Perform vendor validation check and double-check PO matches invoice."
        )

    # Rule 4: Suspicious Vendor/Pattern Rule (MEDIUM)
    if (
        (vendor_id.lower().startswith("susp") or vendor_id.lower() == "unknown")
        and anomaly_score >= settings.RISK_RULE_SUSPICIOUS_VENDOR_ANOMALY_THRESHOLD
    ):
        triggered_rules.append("Suspicious Vendor Pattern (Medium)")
        if highest_severity not in ["CRITICAL", "HIGH"]:
            highest_severity = "MEDIUM"
        mitigation_actions.append(
            "Perform vendor background verification and request secondary approval."
        )

    # Rule 5: High Risk Factor Impact Rule (MEDIUM)
    # Check if 'Amount' is in top risk factors with positive influence
    amount_factor = next((f for f in top_risk_factors if f.get("feature") == "Amount"), None)
    # If the influence is positive, it contributed to making the transaction anomalous
    if (
        amount_factor
        and amount_factor.get("influence", 0.0) > 0.0
        and anomaly_score >= settings.RISK_RULE_HIGH_FACTOR_ANOMALY_THRESHOLD
    ):
        triggered_rules.append("High Risk Factor Impact (Medium)")
        if highest_severity not in ["CRITICAL", "HIGH"]:
            highest_severity = "MEDIUM"
        mitigation_actions.append(
            "Audit transaction posting times and check for split transaction attempts."
        )

    # Rule 6: Blacklisted Vendor Rule (CRITICAL)
    if vendor_risk and vendor_risk.get("is_blacklisted"):
        triggered_rules.append("Blacklisted Vendor (Critical)")
        highest_severity = "CRITICAL"
        mitigation_actions.append(
            "Immediate freeze of transaction. Suspicious blacklisted vendor account."
        )

    # Rule 7: Watchlist Vendor Rule (HIGH)
    if vendor_risk and vendor_risk.get("is_watchlist"):
        triggered_rules.append("Watchlist Vendor (High)")
        if highest_severity != "CRITICAL":
            highest_severity = "HIGH"
        mitigation_actions.append(
            "Route transaction for intensive compliance audit."
        )

    # Rule 8: Low Vendor Reputation Rule (HIGH/MEDIUM)
    if vendor_risk and vendor_risk.get("reputation_score") is not None:
        rep_score = float(vendor_risk["reputation_score"])
        if rep_score < 30.0:
            triggered_rules.append("Low Vendor Reputation (High)")
            if highest_severity not in ["CRITICAL"]:
                highest_severity = "HIGH"
            mitigation_actions.append(
                "Verify invoice details and request executive sign-off."
            )
        elif rep_score < 50.0:
            triggered_rules.append("Low Vendor Reputation (Medium)")
            if highest_severity not in ["CRITICAL", "HIGH"]:
                highest_severity = "MEDIUM"
            mitigation_actions.append(
                "Verify vendor credentials and double-check PO matching."
            )

    # Rule 9: High Vendor Historical Alerts Rule (MEDIUM)
    if vendor_risk and vendor_risk.get("historical_alerts_count", 0) >= 3:
        triggered_rules.append("High Vendor Historical Alerts (Medium)")
        if highest_severity not in ["CRITICAL", "HIGH"]:
            highest_severity = "MEDIUM"
        mitigation_actions.append(
            "Check vendor billing history and alert log patterns."
        )

    # If no rules triggered but anomaly_score >= settings.FRAUD_THRESHOLD, default to MEDIUM
    if not triggered_rules and anomaly_score >= settings.FRAUD_THRESHOLD:
        triggered_rules.append("ML Anomaly Score (Medium)")
        highest_severity = "MEDIUM"
        mitigation_actions.append("Review transaction logs and verify approval signature.")

    # Determine fallback mitigation if none triggered
    if not mitigation_actions:
        mitigation_actions.append("No mitigation actions required for low-risk transaction.")

    # Format unique actions as a single string
    unique_actions = list(dict.fromkeys(mitigation_actions))
    recommended_mitigation = " | ".join(unique_actions)

    return highest_severity, triggered_rules, recommended_mitigation
