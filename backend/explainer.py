"""
explainer.py — AnomalyIQ Reason Codes + LLM Narrative Generator

build_reason_codes(): up to 5 plain-English reasons from:
  - triggered rules
  - graph signals
  - top ML features (Z-score based)

generate_llm_narrative(): calls OpenRouter (nemotron-3-super-120b-a12b:free)
  - Prompt: ≤80 words, no bullets, auditor-ready
  - Fallback narrative on any OpenRouter failure — does NOT surface error to user
"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Self-contained env load — safe to call multiple times (load_dotenv is a no-op if already loaded)
load_dotenv()


# ── Feature name humanization map ────────────────────────────────────────────

FEATURE_NAME_MAP = {
    # Generic model features
    "v1": "anonymized_feature_1",
    "v2": "anonymized_feature_2",
    "v3": "anonymized_feature_3",
    "v4": "anonymized_feature_4",
    "v5": "anonymized_feature_5",
    "v6": "anonymized_feature_6",
    "v7": "anonymized_feature_7",
    "v8": "anonymized_feature_8",
    "v9": "anonymized_feature_9",
    "v10": "anonymized_feature_10",
    "v11": "anonymized_feature_11",
    "v12": "anonymized_feature_12",
    "v13": "anonymized_feature_13",
    "v14": "spending_deviation",
    "v15": "transaction_pattern",
    "v16": "account_activity",
    "v17": "merchant_risk",
    "v18": "geo_anomaly",
    "v19": "time_anomaly",
    "v20": "network_signal",
    "v21": "device_risk",
    "v22": "auth_pattern",
    "v23": "channel_risk",
    "v24": "behavioral_anomaly",
    "v25": "session_risk",
    "v26": "frequency_anomaly",
    "v27": "correlation_signal",
    "v28": "latency_anomaly",
    # Demo dataset features
    "amount": "transaction_amount",
    "hour": "transaction_hour",
    "velocity_1hr": "hourly_velocity",
    "distance_from_home": "geographic_distance",
    "foreign_transaction": "foreign_origin",
    "high_risk_merchant": "merchant_risk_score",
    "deviation_score": "behavioral_deviation",
}


def humanize_feature(name: str) -> str:
    """Convert raw feature name to human-readable version."""
    lower = name.lower()
    if lower in FEATURE_NAME_MAP:
        return FEATURE_NAME_MAP[lower]
    # Auto-humanize: replace underscores and camelCase
    return name.replace("_", " ").replace("-", " ")


def build_reason_codes(
    triggered_rules: list[dict],
    graph_signals: list[str],
    feature_names: list[str],
    raw_values: dict[str, float],
    col_stats: dict[str, dict],
    ml_probabilities: Optional[list[float]] = None,
    max_reasons: int = 5,
) -> list[dict]:
    """
    Build up to 5 plain-English reason codes for a prediction.

    Priority order:
      1. Triggered rules (highest weight first)
      2. Graph signals (if space remains)
      3. Top ML features by Z-score (if space remains)

    Returns:
        list of {code, description, source, severity} dicts
    """
    reasons = []

    # ── 1. Triggered rules ───────────────────────────────────────────────────
    sorted_rules = sorted(triggered_rules, key=lambda r: -r.get("weight", 0))
    for rule in sorted_rules:
        if len(reasons) >= max_reasons:
            break
        reasons.append({
            "code": rule.get("id", "RULE"),
            "description": rule.get("description", rule.get("name", "Policy rule triggered")),
            "source": "rule_engine",
            "severity": "high" if rule.get("weight", 0) >= 0.25 else "medium",
        })

    # ── 2. Graph signals ─────────────────────────────────────────────────────
    for signal in graph_signals:
        if len(reasons) >= max_reasons:
            break
        if isinstance(signal, str) and len(signal) > 10:
            reasons.append({
                "code": "GRAPH_SIGNAL",
                "description": signal,
                "source": "graph_engine",
                "severity": "medium",
            })

    # ── 3. Top ML features by Z-score ────────────────────────────────────────
    if len(reasons) < max_reasons:
        z_scores = []
        for feat in feature_names:
            if feat not in raw_values or feat not in col_stats:
                continue
            stats = col_stats[feat]
            val = float(raw_values.get(feat, 0))
            mean = float(stats.get("mean", 0))
            std = float(stats.get("std", 1)) or 1.0
            z = (val - mean) / std
            z_scores.append((feat, val, z, mean, std))

        # Sort by absolute Z-score descending
        z_scores.sort(key=lambda x: -abs(x[2]))

        for feat, val, z, mean, std in z_scores:
            if len(reasons) >= max_reasons:
                break
            if abs(z) < 1.5:
                continue  # Not anomalous enough to report

            human_feat = humanize_feature(feat)
            direction = "elevated above" if z > 0 else "significantly below"
            magnitude = "significantly" if abs(z) > 3 else "notably"

            reasons.append({
                "code": f"ML_FEATURE_{feat.upper()}",
                "description": (
                    f"{human_feat.replace('_', ' ').title()} is {magnitude} {direction} "
                    f"expected range (Z-score: {z:+.1f})"
                ),
                "source": "ml_model",
                "severity": "high" if abs(z) > 3 else "medium",
            })

    # If no reasons at all, provide a generic fallback
    if not reasons:
        reasons.append({
            "code": "BASELINE_CHECK",
            "description": "Transaction profile within expected parameters — no specific anomaly signals detected",
            "source": "system",
            "severity": "low",
        })

    return reasons[:max_reasons]


def generate_llm_narrative(
    final_score: float,
    risk_band: str,
    verdict: str,
    reason_codes: list[dict],
    ml_score: float,
    rule_score: float,
    graph_score: float,
    flags: int,
    feature_summary: Optional[str] = None,
    api_key: Optional[str] = None,
    session_context: Optional[dict] = None,
) -> dict:
    """
    Generate an auditor-ready LLM narrative via OpenRouter.
    Falls back silently to a local template on any failure.

    Args:
        session_context: {dataset_name, domain_description} from session_context.json

    Returns:
        {narrative: str, source: "openrouter" | "local_fallback"}
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY")

    ctx = session_context or {}
    domain = ctx.get("domain_description", "Anomaly detection")
    dataset = ctx.get("dataset_name", "Unnamed Dataset")

    if api_key:
        try:
            return _call_openrouter(
                api_key=api_key,
                final_score=final_score,
                risk_band=risk_band,
                verdict=verdict,
                reason_codes=reason_codes,
                ml_score=ml_score,
                rule_score=rule_score,
                graph_score=graph_score,
                flags=flags,
                domain=domain,
                dataset=dataset,
            )
        except Exception as e:
            print(f"OpenRouter API Error (llm_narrative): {e}")
            pass  # Fall through to local fallback

    # ── Local fallback ───────────────────────────────────────────────────────
    return {
        "narrative": _local_fallback_narrative(
            final_score=final_score,
            risk_band=risk_band,
            verdict=verdict,
            reason_codes=reason_codes,
            ml_score=ml_score,
            rule_score=rule_score,
            graph_score=graph_score,
            flags=flags,
        ),
        "source": "local_fallback"
    }


def _call_openrouter(
    api_key: str,
    final_score: float,
    risk_band: str,
    verdict: str,
    reason_codes: list[dict],
    ml_score: float,
    rule_score: float,
    graph_score: float,
    flags: int,
    domain: str = "Anomaly detection",
    dataset: str = "Unnamed Dataset",
) -> dict:
    """Call OpenRouter with nemotron model for LLM narrative."""
    from openai import OpenAI

    reasons_text = "; ".join(r["description"] for r in reason_codes[:3])

    prompt = (
        f"Write a concise fraud analysis for a compliance officer (max 80 words, no bullets). "
        f"Transaction risk: {risk_band} ({final_score*100:.0f}%). "
        f"Verdict: {verdict}. "
        f"ML ensemble: {ml_score*100:.0f}%, Rule engine: {rule_score*100:.0f}%, "
        f"Graph intelligence: {graph_score*100:.0f}%. "
        f"{flags} of 4 models flagged as fraud. "
        f"Key signals: {reasons_text}. "
        f"Be professional, specific, and actionable."
    )

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    response = client.chat.completions.create(
        model="nvidia/nemotron-3-super-120b-a12b:free",
        messages=[
            {
                "role": "system",
                "content": (
                    f"You are a financial forensics AI assistant specializing in {domain}. "
                    f"The dataset being analyzed is: {dataset}. "
                    f"Write brief, professional, auditor-ready fraud analysis narratives. "
                    f"Be concise — never exceed 100 words. No bullet points."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=768,
        temperature=0.2,
    )

    narrative = response.choices[0].message.content.strip()
    return {"narrative": narrative, "source": "openrouter"}


def _local_fallback_narrative(
    final_score: float,
    risk_band: str,
    verdict: str,
    reason_codes: list[dict],
    ml_score: float,
    rule_score: float,
    graph_score: float,
    flags: int,
) -> str:
    """Generate a structured local fallback narrative when OpenRouter is unavailable."""
    score_pct = final_score * 100
    ml_pct = ml_score * 100
    rule_pct = rule_score * 100
    graph_pct = graph_score * 100

    if risk_band == "HIGH":
        stance = "This transaction presents a high likelihood of fraud and should be immediately escalated for manual review."
        action = "Recommend blocking pending investigation."
    elif risk_band == "MEDIUM":
        stance = "This transaction exhibits moderate risk indicators warranting additional verification steps."
        action = "Recommend stepped-up authentication before processing."
    else:
        stance = "This transaction falls within acceptable risk parameters and is consistent with normal activity patterns."
        action = "Recommend standard processing."

    top_reason = reason_codes[0]["description"] if reason_codes else "No specific anomaly signals detected"

    return (
        f"Composite risk score of {score_pct:.0f}% ({risk_band}) derived from ML ensemble ({ml_pct:.0f}%), "
        f"rule engine ({rule_pct:.0f}%), and graph intelligence ({graph_pct:.0f}%). "
        f"{flags} of 4 machine learning models flagged this transaction as fraudulent. "
        f"Primary signal: {top_reason}. "
        f"{stance} {action}"
    )


def generate_insight_report(
    metrics: dict,
    col_stats: dict,
    feature_names: list[str],
    n_samples: int,
    smote_applied: bool,
    api_key: Optional[str] = None,
    session_context: Optional[dict] = None,
) -> dict:
    """
    Generate a plain-English insight report for the trained model session.
    Used by GET /api/insight_report.

    Args:
        session_context: {dataset_name, domain_description} from session_context.json
    """
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY", "")

    ctx = session_context or {}
    domain = ctx.get("domain_description", "Anomaly detection")
    dataset = ctx.get("dataset_name", "Unnamed Dataset")

    # Find best model by AUC
    best_model = None
    best_auc = 0.0
    for model_name, model_metrics in metrics.items():
        auc = model_metrics.get("auc", 0)
        if auc > best_auc:
            best_auc = auc
            best_model = model_name

    # Top feature by importance (decision tree) or coefficient (LR)
    top_feature = "N/A"
    top_feature_score = 0.0
    if "decision_tree" in metrics:
        fi = metrics["decision_tree"].get("feature_importances", {})
        if fi:
            top_feature = max(fi, key=fi.get)
            top_feature_score = fi[top_feature]
    if top_feature == "N/A" and "logistic_regression" in metrics:
        coef = metrics["logistic_regression"].get("coefficients", {})
        if coef:
            top_feature = max(coef, key=coef.get)
            top_feature_score = coef[top_feature]

    # Try LLM summary
    narrative = None
    source = "local_fallback"

    if api_key:
        try:
            prompt = (
                f"Summarize this fraud detection model session for a compliance officer (max 80 words, no bullets). "
                f"Dataset: {dataset} ({n_samples} samples). "
                f"Domain: {domain}. "
                f"Best model: {best_model} (AUC {best_auc:.3f}). "
                f"SMOTE resampling: {'applied' if smote_applied else 'not needed'}. "
                f"Top predictive feature: {humanize_feature(top_feature)} (importance {top_feature_score:.3f}). "
                f"Be professional and highlight actionable insights."
            )

            from openai import OpenAI
            client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=api_key,
            )
            response = client.chat.completions.create(
                model="nvidia/nemotron-3-super-120b-a12b:free",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            f"You are a financial forensics AI assistant specializing in {domain}. "
                            f"Write brief, professional model performance summaries."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                max_tokens=250,
                temperature=0.3,
            )
            narrative = response.choices[0].message.content.strip()
            source = "openrouter"
        except Exception as e:
            print(f"OpenRouter API Error (insight_report): {e}")
            pass

    if narrative is None:
        narrative = (
            f"The fraud detection model ensemble has been successfully trained on {n_samples} samples. "
            f"The best performing model is {best_model.replace('_', ' ').title() if best_model else 'the ensemble'} "
            f"with an AUC of {best_auc:.3f}, indicating {'excellent' if best_auc > 0.9 else 'strong' if best_auc > 0.8 else 'moderate'} "
            f"discriminative power. "
            f"{'SMOTE oversampling was applied to address class imbalance. ' if smote_applied else ''}"
            f"The most influential predictive feature is {humanize_feature(top_feature).replace('_', ' ')} "
            f"(importance score: {top_feature_score:.3f}). "
            f"Review the model comparison table and ROC curves for detailed performance metrics."
        )

    return {
        "narrative": narrative,
        "source": source,
        "best_model": best_model,
        "best_auc": round(best_auc, 4),
        "top_feature": top_feature,
        "top_feature_human": humanize_feature(top_feature),
        "top_feature_score": round(top_feature_score, 4),
        "n_samples": n_samples,
        "smote_applied": smote_applied,
    }


def generate_cluster_narrative(
    cluster_payload: dict,
    session_context: Optional[dict] = None,
    api_key: Optional[str] = None,
) -> str:
    """Generate LLM narrative for an entire fraud ring/cluster."""
    if api_key is None:
        api_key = os.getenv("OPENROUTER_API_KEY", "")

    ctx = session_context or {}
    domain = ctx.get("domain_description", "Anomaly detection")

    transactions = cluster_payload.get("transactions", [])
    shared = cluster_payload.get("shared_entities", [])
    
    # Format data for prompt
    tx_count = len(transactions)
    fraud_count = cluster_payload.get("fraud_count", 0)
    hub_count = cluster_payload.get("hub_count", 0)
    
    shared_str = ", ".join([f"{s['feature'].replace('_', ' ')} ({s['bucket']})" for s in shared[:5]])
    if not shared_str:
        shared_str = "complex network topology"
    
    if api_key:
        try:
            prompt = (
                f"Analyze this fraud ring for a human investigator in {domain} (max 80 words, no bullets). "
                f"Cluster contains {tx_count} connected accounts, {fraud_count} flagged as fraud. "
                f"{hub_count} accounts act as central hubs. "
                f"They are strongly linked by sharing: {shared_str}. "
                f"Explain why this looks like a coordinated attack and what the shared entities imply. Be actionable."
            )
            from openai import OpenAI
            client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=api_key)
            response = client.chat.completions.create(
                model="nvidia/nemotron-3-super-120b-a12b:free",
                messages=[
                    {"role": "system", "content": "You are a senior financial forensics AI analyst. Write professional, auditor-ready fraud ring summaries. Never use bullets."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=150, temperature=0.3
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenRouter API Error (cluster_narrative): {e}")
            pass

    # Fallback
    return (
        f"Fraud Ring detected involving {tx_count} interacting accounts, with {fraud_count} anomalous transactions. "
        f"This cluster forms a coordinated network linked primarily by shared attributes: {shared_str}. "
        f"{hub_count} accounts are identified as central hubs. Recommend immediate blocking and investigation of the shared entities."
    )

