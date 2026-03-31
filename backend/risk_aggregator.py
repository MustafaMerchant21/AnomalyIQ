"""
risk_aggregator.py — AnomalyIQ Weighted Risk Score Fusion

Final Score = 0.40 × ML Score
            + 0.30 × Rule Score
            + 0.30 × Graph Score

Risk Bands:
  0.00 – 0.40  →  LOW     #3DDC97
  0.40 – 0.70  →  MEDIUM  #F5A623
  0.70 – 1.00  →  HIGH    #FF4081

Confidence:
  disagreement < 20 AND flags in (0,4)  →  HIGH
  disagreement < 45                     →  MEDIUM
  else                                  →  LOW
"""


def aggregate_risk(
    ml_score: float,
    rule_score: float,
    graph_score: float,
    disagreement_index: float,
    flags: int,
) -> dict:
    """
    Compute the final composite risk score and risk band.

    Args:
        ml_score: ML ensemble probability [0, 1]
        rule_score: Rule engine score [0, 1]
        graph_score: Graph intelligence score [0, 1]
        disagreement_index: Model disagreement metric [0, 100]
        flags: Number of models flagging as fraud [0, 4]

    Returns:
        {
            final_score: float [0, 1],
            risk_band: "LOW" | "MEDIUM" | "HIGH",
            risk_color: hex color string,
            confidence: "HIGH" | "MEDIUM" | "LOW",
            score_breakdown: {ml, rule, graph, weighted_ml, weighted_rule, weighted_graph}
        }
    """
    # Clamp inputs to [0, 1]
    ml_score = max(0.0, min(1.0, float(ml_score)))
    rule_score = max(0.0, min(1.0, float(rule_score)))
    graph_score = max(0.0, min(1.0, float(graph_score)))

    # Weighted fusion
    weighted_ml = 0.40 * ml_score
    weighted_rule = 0.30 * rule_score
    weighted_graph = 0.30 * graph_score

    final_score = max(0.0, min(1.0, weighted_ml + weighted_rule + weighted_graph))

    # Risk band classification
    if final_score < 0.40:
        risk_band = "LOW"
        risk_color = "#3DDC97"
    elif final_score < 0.70:
        risk_band = "MEDIUM"
        risk_color = "#F5A623"
    else:
        risk_band = "HIGH"
        risk_color = "#FF4081"

    # Confidence assessment
    disagreement = max(0.0, float(disagreement_index))
    if disagreement < 20 and flags in (0, 4):
        confidence = "HIGH"
    elif disagreement < 45:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    return {
        "final_score": round(final_score, 4),
        "final_score_pct": round(final_score * 100, 1),
        "risk_band": risk_band,
        "risk_color": risk_color,
        "confidence": confidence,
        "score_breakdown": {
            "ml_score": round(ml_score, 4),
            "rule_score": round(rule_score, 4),
            "graph_score": round(graph_score, 4),
            "weighted_ml": round(weighted_ml, 4),
            "weighted_rule": round(weighted_rule, 4),
            "weighted_graph": round(weighted_graph, 4),
        }
    }


def compute_ml_ensemble_score(probabilities: list[float]) -> float:
    """
    Compute the ensemble ML score from per-model probabilities.
    Uses the mean of available model probabilities.
    """
    valid = [p for p in probabilities if p is not None and not _is_nan(p)]
    if not valid:
        return 0.0
    return sum(valid) / len(valid)


def _is_nan(val) -> bool:
    try:
        import math
        return math.isnan(float(val))
    except (TypeError, ValueError):
        return False
