"""
rule_engine.py — AnomalyIQ Threshold & Policy Rule Engine

Rules fire based on feature value heuristics using case-insensitive column name matching.
No rigid schema required — matches column names containing key substrings.

Rules:
  HIGH_AMOUNT      (w=0.30) — amount > mean + 3×std
  VELOCITY_BURST   (w=0.25) — velocity/burst feature > 4
  FOREIGN_LOCATION (w=0.20) — location/geo/foreign feature > 0.5
  NEW_DEVICE       (w=0.15) — device/ip feature > 0.5
  UNUSUAL_TIME     (w=0.10) — hour column between 1–5 AM
"""

from typing import Any, Optional


# ── Rule definitions ─────────────────────────────────────────────────────────

RULES = [
    {
        "id": "HIGH_AMOUNT",
        "name": "High Transaction Amount",
        "weight": 0.30,
        "description": "Transaction amount significantly exceeds historical norms (>3σ above mean)",
        "column_keywords": ["amount", "value", "sum", "total", "price"],
        "type": "statistical",  # requires col_stats
    },
    {
        "id": "VELOCITY_BURST",
        "name": "Transaction Velocity Burst",
        "weight": 0.25,
        "description": "Multiple rapid transactions detected in a short time window",
        "column_keywords": ["velocity", "burst", "freq", "count", "rate"],
        "type": "threshold",
        "threshold": 4,
    },
    {
        "id": "FOREIGN_LOCATION",
        "name": "Foreign/Unusual Location",
        "weight": 0.20,
        "description": "Transaction originates from a foreign or unusual geographic location",
        "column_keywords": ["foreign", "location", "geo", "country", "overseas", "international", "abroad"],
        "type": "threshold",
        "threshold": 0.5,
    },
    {
        "id": "NEW_DEVICE",
        "name": "Unrecognized Device/IP",
        "weight": 0.15,
        "description": "Transaction initiated from an unrecognized device or IP address",
        "column_keywords": ["device", "ip", "fingerprint", "browser", "agent", "new_device", "high_risk"],
        "type": "threshold",
        "threshold": 0.5,
    },
    {
        "id": "UNUSUAL_TIME",
        "name": "Unusual Transaction Time",
        "weight": 0.10,
        "description": "Transaction occurred during off-hours (1–5 AM), a common fraud window",
        "column_keywords": ["hour", "time", "timestamp", "hr"],
        "type": "time_range",
        "low": 1,
        "high": 5,
    },
    {
        "id": "AT_RISK_CATEGORY",
        "name": "High-Risk Categories",
        "weight": 0.20,
        "description": "Transaction involves a merchant category historically tied to fraud",
        "column_keywords": ["category", "type", "mcc", "merchant_category"],
        "type": "categorical_match",
        "high_risk_values": ["crypto", "gaming", "jewelry", "electronics", "transfer", "cash", "luxury", "gambling"],
    },
    {
        "id": "ELDERLY_TARGET",
        "name": "Elderly Demographic Target",
        "weight": 0.15,
        "description": "Customer age >= 65, a demographic frequently targeted by scams",
        "column_keywords": ["age_years", "age"],
        "type": "threshold",
        "threshold": 65.0,
    },
]


def _find_column(feature_names: list[str], keywords: list[str]) -> Optional[str]:
    """Find first column whose name contains any keyword (case-insensitive)."""
    feature_lower = {f.lower(): f for f in feature_names}
    for keyword in keywords:
        for col_lower, col_orig in feature_lower.items():
            if keyword in col_lower:
                return col_orig
    return None


def evaluate_rules(
    feature_values: dict[str, Any],
    col_stats: dict[str, dict],
    feature_names: Optional[list[str]] = None
) -> dict:
    """
    Evaluate all rules for a single transaction.

    Args:
        feature_values: dict mapping feature name → raw (unscaled) value
        col_stats: dict mapping feature name → {min, max, mean, std}
        feature_names: optional ordered list of feature names

    Returns:
        {
            rule_score: float [0, 1],
            triggered_rules: list of rule dicts,
            rules_fired: int,
            rule_details: list of all rules with fired bool
        }
    """
    if feature_names is None:
        feature_names = list(feature_values.keys())

    triggered = []
    all_rule_details = []

    for rule in RULES:
        col = _find_column(feature_names, rule["column_keywords"])
        fired = False
        value = None
        description = rule["description"]

        if col is not None and col in feature_values:
            value = feature_values[col]

            if rule["type"] == "statistical":
                stats = col_stats.get(col, {})
                mean = stats.get("mean", 0)
                std = stats.get("std", 1)
                threshold = mean + 3 * std
                fired = float(value) > threshold
                description = (
                    f"Amount ${value:.2f} exceeds threshold ${threshold:.2f} "
                    f"(mean ${mean:.2f} + 3×σ ${std:.2f})"
                )

            elif rule["type"] == "threshold":
                try:
                    val_float = float(value)
                    fired = val_float > rule["threshold"]
                    description = (
                        f"{col} = {val_float:.3f} exceeds threshold {rule['threshold']}"
                    )
                except (TypeError, ValueError):
                    pass

            elif rule["type"] == "time_range":
                try:
                    val_float = float(value)
                    fired = rule["low"] <= val_float <= rule["high"]
                    description = (
                        f"Transaction at {int(val_float):02d}:00 — within off-hours window "
                        f"({rule['low']}–{rule['high']} AM)"
                    )
                except (TypeError, ValueError):
                    pass

            elif rule["type"] == "categorical_match":
                val_str = str(value).lower()
                matched = [k for k in rule["high_risk_values"] if k in val_str]
                if matched:
                    fired = True
                    description = f"Category '{value}' is flagged as high-risk (matched: {', '.join(matched)})"

        rule_detail = {
            "id": rule["id"],
            "name": rule["name"],
            "weight": rule["weight"],
            "fired": fired,
            "column": col,
            "value": value,
            "description": description if fired else rule["description"]
        }
        all_rule_details.append(rule_detail)

        if fired:
            triggered.append(rule_detail)

    rule_score = min(1.0, sum(r["weight"] for r in triggered))

    return {
        "rule_score": rule_score,
        "triggered_rules": triggered,
        "rules_fired": len(triggered),
        "rule_details": all_rule_details
    }
