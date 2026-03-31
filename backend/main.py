"""
main.py — AnomalyIQ FastAPI Application

All endpoints are fully implemented — no stubs, no TODOs.
CORS: allow_origins=["*"]
Session storage: OS temp dir (see session_manager.py)

Run with: uvicorn main:app --reload --port 8000
"""

import os
import io
import uuid
import json
import csv
import time
import threading
import traceback
import tempfile
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Optional, Any
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, UploadFile, Form, Query, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel

from dotenv import load_dotenv
from pathlib import Path as _Path

# Load .env relative to this file's location, not the CWD.
# This ensures the key is found regardless of where uvicorn is launched from.
load_dotenv(dotenv_path=_Path(__file__).parent / ".env")

from session_manager import (
    ensure_session_dir, save_json, load_json, save_training_status,
    load_training_status, session_exists, cleanup_old_sessions,
    get_session_file_path, _json_serializer, load_session_context
)
from demo_generator import get_demo_csv_bytes
from rule_engine import evaluate_rules
from graph_engine import run_graph_analysis
from risk_aggregator import aggregate_risk, compute_ml_ensemble_score
from explainer import build_reason_codes, generate_llm_narrative, generate_insight_report
import joblib


# ── Application Startup ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: cleanup old sessions."""
    cleanup_old_sessions(max_sessions=20)
    yield


app = FastAPI(
    title="AnomalyIQ API",
    version="2.0.0",
    description="AI-powered fraud detection system",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Active training threads guard ─────────────────────────────────────────────
_training_locks: dict[str, threading.Lock] = {}


# ── Pydantic Models ────────────────────────────────────────────────────────────

class PredictRequest(BaseModel):
    session_id: str
    features: dict[str, Any]

class RiskScoreRequest(BaseModel):
    session_id: str
    features: dict[str, Any]

class ExplainRequest(BaseModel):
    session_id: str
    features: dict[str, Any]
    ml_result: Optional[dict] = None
    rule_result: Optional[dict] = None
    graph_result: Optional[dict] = None

class CaseSaveRequest(BaseModel):
    session_id: str
    transaction_id: str
    decision: str  # flag | allow | escalate | note
    note: Optional[str] = None
    transaction_data: Optional[dict] = None
    risk_score: Optional[float] = None
    verdict: Optional[str] = None

class InjectPointRequest(BaseModel):
    session_id: str
    features: dict[str, Any]


# ── Helper functions ──────────────────────────────────────────────────────────

def _require_session(session_id: Optional[str]) -> str:
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")
    if not session_exists(session_id):
        raise HTTPException(status_code=404, detail=f"Session '{session_id}' not found")
    return session_id


def _require_metrics(session_id: str) -> dict:
    metrics = load_json(session_id, "metrics.json")
    if not metrics:
        raise HTTPException(status_code=404, detail="Model not trained yet — run /api/upload first")
    return metrics


def _require_column_config(session_id: str) -> dict:
    cfg = load_json(session_id, "column_config.json")
    if not cfg:
        raise HTTPException(status_code=404, detail="Column config not found")
    return cfg


def _scale_features(features: dict, session_id: str, cfg: dict) -> tuple[np.ndarray, list[str]]:
    """
    Transform raw mixed-type feature values using the session's ColumnTransformer.
    (Keeps the old function name for API compatibility, but uses the new preprocessor).
    """
    try:
        from preprocessor import transform_single_row
        return transform_single_row(features, session_id)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


def _run_training_thread(session_id: str, df: pd.DataFrame, feature_cols: list, target_col: str):
    """Run the full training pipeline in a background thread."""
    try:
        from pipeline import run_full_pipeline
        run_full_pipeline(
            df=df,
            feature_cols=feature_cols,
            target_col=target_col,
            session_id=session_id,
        )

        # Generate insight report asynchronously after training
        try:
            metrics = load_json(session_id, "metrics.json")
            cfg = load_json(session_id, "column_config.json")
            ctx = load_session_context(session_id)
            if metrics and cfg:
                report = generate_insight_report(
                    metrics=metrics,
                    col_stats=cfg.get("col_stats", {}),
                    feature_names=cfg.get("feature_cols", []),
                    n_samples=cfg.get("n_total", 0),
                    smote_applied=cfg.get("smote_applied", False),
                    api_key=os.getenv("OPENROUTER_API_KEY", ""),
                    session_context=ctx,
                )
                save_json(session_id, "insight_report.json", report)
        except Exception as e:
            print("Insight report generation error:", repr(e))

        # Run graph analysis on test samples
        try:
            test_samples = load_json(session_id, "test_samples.json")
            if test_samples:
                metrics = load_json(session_id, "metrics.json")
                cfg = load_json(session_id, "column_config.json")

                # Add anomaly scores from best model
                from pipeline import load_model, load_scaler
                scaler = load_scaler(session_id)
                best_model = None
                best_auc = 0
                if metrics:
                    for mn, mv in metrics.items():
                        if mv.get("auc", 0) > best_auc:
                            best_auc = mv["auc"]
                            best_model_name = mn

                    try:
                        best_model = load_model(session_id, best_model_name)
                    except Exception:
                        pass

                scored_samples = []
                for s in test_samples:
                    feature_cols_l = cfg.get("feature_cols", []) if cfg else []
                    vals = np.array([s.get(f, 0.0) for f in feature_cols_l]).reshape(1, -1)
                    score = 0.0
                    if best_model is not None:
                        try:
                            score = float(best_model.predict_proba(vals)[:, 1][0])
                        except Exception:
                            pass
                    scored_samples.append({
                        **s,
                        "anomaly_score": score,
                        "_true_label": s.get("_true_label", 0)
                    })

                graph_result = run_graph_analysis(scored_samples[:500])  # Limit for performance
                save_json(session_id, "graph_analysis.json", graph_result)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("Graph analysis crash:", repr(e))

    except Exception as e:
        error_msg = str(e)
        save_training_status(session_id, 0, f"Training failed: {error_msg}", done=True, error=error_msg)


# ─────────────────────────────────────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def health():
    """Health check endpoint."""
    return {"status": "ok", "version": "2.0.0"}


@app.post("/api/upload")
async def upload_csv(
    file: UploadFile = File(...),
    feature_cols: str = Form(...),
    target_col: str = Form(...),
    dataset_name: str = Form(default="Unnamed Dataset"),
    domain_description: str = Form(default="Anomaly detection"),
):
    """
    Upload a CSV file for fraud detection training.
    feature_cols: comma-separated list of feature column names
    target_col: name of the target/label column
    dataset_name: optional human-readable name for the dataset (used in LLM prompts)
    domain_description: optional domain context (e.g. 'credit card fraud for retail banking')
    """
    session_id = str(uuid.uuid4())
    ensure_session_dir(session_id)

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    feature_list = [c.strip() for c in feature_cols.split(",") if c.strip()]

    # Validate columns exist
    missing = [c for c in feature_list + [target_col] if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Columns not found in CSV: {missing}")

    # Save session context immediately — available to all LLM calls
    import datetime
    save_json(session_id, "session_context.json", {
        "dataset_name": dataset_name.strip() or "Unnamed Dataset",
        "domain_description": domain_description.strip() or "Anomaly detection",
        "uploaded_at": datetime.datetime.utcnow().isoformat() + "Z",
    })

    save_training_status(session_id, 0, "Queued for training", done=False)

    thread = threading.Thread(
        target=_run_training_thread,
        args=(session_id, df, feature_list, target_col),
        daemon=True,
    )
    thread.start()

    return {"session_id": session_id, "message": "Training started", "rows": len(df)}


@app.post("/api/upload_demo")
async def upload_demo():
    """
    Train on synthetically generated demo dataset (2000 rows, ~8% fraud).
    No file upload required.
    """
    session_id = str(uuid.uuid4())
    ensure_session_dir(session_id)

    csv_bytes = get_demo_csv_bytes()
    df = pd.read_csv(io.BytesIO(csv_bytes), low_memory=False)

    feature_cols = [
        "cc_num", "name", "amount", "hour", "velocity_1hr", "distance_from_home",
        "foreign_transaction", "high_risk_merchant", "deviation_score"
    ]
    target_col = "is_fraud"

    save_training_status(session_id, 0, "Queued for training", done=False)

    thread = threading.Thread(
        target=_run_training_thread,
        args=(session_id, df, feature_cols, target_col),
        daemon=True,
    )
    thread.start()

    return {"session_id": session_id, "message": "Demo training started", "rows": len(df)}


@app.get("/api/training_status")
async def training_status(session_id: str = Query(...)):
    """Poll training progress. Returns step (1-9), message, done, error."""
    _require_session(session_id)
    status = load_training_status(session_id)
    if not status:
        return {"step": 0, "message": "Initializing...", "done": False, "error": None}
    return status


@app.get("/api/metrics")
async def get_metrics(session_id: str = Query(...)):
    """Return full metrics.json for the session (all 4 models)."""
    _require_session(session_id)
    metrics = _require_metrics(session_id)
    return metrics


@app.get("/api/confusion_matrix")
async def get_confusion_matrix(
    session_id: str = Query(...),
    model: str = Query(default="logistic_regression"),
):
    """Return confusion matrix for a specific model."""
    _require_session(session_id)
    metrics = _require_metrics(session_id)

    if model not in metrics:
        valid = list(metrics.keys())
        raise HTTPException(status_code=400, detail=f"Model '{model}' not found. Valid: {valid}")

    cm = metrics[model].get("confusion_matrix", {"tn": 0, "fp": 0, "fn": 0, "tp": 0})
    return {"model": model, "confusion_matrix": cm}


@app.get("/api/roc_curve")
async def get_roc_curve(session_id: str = Query(...)):
    """Return ROC curves for all 4 models."""
    _require_session(session_id)
    metrics = _require_metrics(session_id)

    result = {}
    for model_name, model_data in metrics.items():
        result[model_name] = model_data.get("roc_curve", {"fpr": [], "tpr": []})
    return result


@app.get("/api/pr_curve")
async def get_pr_curve(session_id: str = Query(...)):
    """Return Precision-Recall curves for all 4 models."""
    _require_session(session_id)
    metrics = _require_metrics(session_id)

    result = {}
    for model_name, model_data in metrics.items():
        result[model_name] = model_data.get("pr_curve", {"precision": [], "recall": []})
    return result


@app.get("/api/column_config")
async def get_column_config(session_id: str = Query(...)):
    """Return feature names, target, col_stats, class_means, tf_available."""
    _require_session(session_id)
    cfg = _require_column_config(session_id)
    return cfg


@app.get("/api/feature_importance")
async def get_feature_importance(session_id: str = Query(...)):
    """Return DT feature importances + LR coefficients."""
    _require_session(session_id)
    metrics = _require_metrics(session_id)

    result = {
        "feature_importances": metrics.get("decision_tree", {}).get("feature_importances", {}),
        "coefficients": metrics.get("logistic_regression", {}).get("coefficients", {}),
    }
    return result


@app.get("/api/insight_report")
async def get_insight_report(session_id: str = Query(...)):
    """Return cached AI plain-English insight report (generated async after training)."""
    _require_session(session_id)

    report = load_json(session_id, "insight_report.json")
    if report:
        return report

    # Generate on-demand if not cached
    metrics = load_json(session_id, "metrics.json")
    cfg = load_json(session_id, "column_config.json")
    if not metrics or not cfg:
        raise HTTPException(status_code=404, detail="Metrics not available yet")

    report = generate_insight_report(
        metrics=metrics,
        col_stats=cfg.get("col_stats", {}),
        feature_names=cfg.get("feature_cols", []),
        n_samples=cfg.get("n_total", 0),
        smote_applied=cfg.get("smote_applied", False),
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
    )
    save_json(session_id, "insight_report.json", report)
    return report


@app.post("/api/predict")
async def predict(req: PredictRequest):
    """
    Run ML prediction on all 4 models.
    Returns probabilities, flags, verdict, disagreement_index.
    """
    _require_session(req.session_id)
    cfg = _require_column_config(req.session_id)

    X_scaled, feature_cols = _scale_features(req.features, req.session_id, cfg)

    from pipeline import predict_all_models
    result = predict_all_models(X_scaled, req.session_id)
    return result


@app.post("/api/risk_score")
async def risk_score(req: RiskScoreRequest):
    """
    Full composite scoring: ML + Rule + Graph + Aggregation.
    Returns complete risk object with all scores and breakdown.
    """
    _require_session(req.session_id)
    cfg = _require_column_config(req.session_id)

    X_scaled, feature_cols = _scale_features(req.features, req.session_id, cfg)
    col_stats = cfg.get("col_stats", {})

    # ── ML Score ──────────────────────────────────────────────────────────────
    from pipeline import predict_all_models
    ml_result = predict_all_models(X_scaled, req.session_id)

    # ── Rule Score ────────────────────────────────────────────────────────────
    rule_result = evaluate_rules(
        feature_values=req.features,
        col_stats=col_stats,
        feature_names=feature_cols,
    )

    # ── Graph Score ───────────────────────────────────────────────────────────
    # Use pre-computed graph analysis from training
    graph_result = load_json(req.session_id, "graph_analysis.json")
    if not graph_result:
        graph_result = {
            "graph_score": 0.1,
            "signals": ["Graph analysis not yet available — train model first"],
            "pagerank_score": 0.0,
            "connected_anomalies": 0,
            "total_nodes": 0,
            "total_edges": 0,
            "hub_count": 0,
        }

    # ── Aggregate ─────────────────────────────────────────────────────────────
    aggregated = aggregate_risk(
        ml_score=ml_result["ensemble_score"],
        rule_score=rule_result["rule_score"],
        graph_score=graph_result["graph_score"],
        disagreement_index=ml_result["disagreement_index"],
        flags=ml_result["flags"],
    )

    return {
        "ml": ml_result,
        "rules": rule_result,
        "graph": graph_result,
        "risk": aggregated,
    }


@app.post("/api/explain")
async def explain(req: ExplainRequest):
    """
    Generate reason codes + LLM narrative for a transaction.
    """
    _require_session(req.session_id)
    cfg = _require_column_config(req.session_id)

    feature_cols = cfg.get("feature_cols", [])
    col_stats = cfg.get("col_stats", {})

    # Build reason codes
    triggered_rules = req.rule_result.get("triggered_rules", []) if req.rule_result else []
    graph_signals = req.graph_result.get("signals", []) if req.graph_result else []

    ml_probs = []
    if req.ml_result:
        ml_probs = [v for v in req.ml_result.get("probabilities", {}).values() if v is not None]

    reason_codes = build_reason_codes(
        triggered_rules=triggered_rules,
        graph_signals=graph_signals,
        feature_names=feature_cols,
        raw_values=req.features,
        col_stats=col_stats,
        ml_probabilities=ml_probs,
    )

    # Get risk info
    final_score = 0.5
    risk_band = "MEDIUM"
    verdict = "AMBIGUOUS"
    flags = 2
    ml_score = 0.5
    rule_score = 0.0
    graph_score = 0.0
    disagreement = 0.0

    if req.ml_result:
        ml_score = req.ml_result.get("ensemble_score", 0.5)
        flags = req.ml_result.get("flags", 2)
        verdict = req.ml_result.get("verdict", "AMBIGUOUS")
        disagreement = req.ml_result.get("disagreement_index", 0.0)
    if req.rule_result:
        rule_score = req.rule_result.get("rule_score", 0.0)
    if req.graph_result:
        graph_score = req.graph_result.get("graph_score", 0.0)

    agg = aggregate_risk(ml_score, rule_score, graph_score, disagreement, flags)
    final_score = agg["final_score"]
    risk_band = agg["risk_band"]

    narrative_result = generate_llm_narrative(
        final_score=final_score,
        risk_band=risk_band,
        verdict=verdict,
        reason_codes=reason_codes,
        ml_score=ml_score,
        rule_score=rule_score,
        graph_score=graph_score,
        flags=flags,
        api_key=os.getenv("OPENROUTER_API_KEY", ""),
    )

    return {
        "reason_codes": reason_codes,
        "narrative": narrative_result["narrative"],
        "narrative_source": narrative_result["source"],
        "final_score": final_score,
        "risk_band": risk_band,
    }


@app.get("/api/simulate")
async def simulate(session_id: str = Query(...)):
    """
    Return a random test sample with all 4 model predictions and consensus.
    """
    _require_session(session_id)
    cfg = _require_column_config(session_id)
    test_samples = load_json(session_id, "test_samples.json")

    if not test_samples:
        raise HTTPException(status_code=404, detail="No test samples available")

    # Pick random sample
    rng = np.random.default_rng()
    idx = int(rng.integers(0, len(test_samples)))
    sample = test_samples[idx]

    feature_cols = cfg.get("feature_cols", [])
    values = np.array([sample.get(f, 0.0) for f in feature_cols]).reshape(1, -1)

    from pipeline import predict_all_models
    ml_result = predict_all_models(values, session_id)

    raw_values = sample.get("_raw_values", {})
    true_label = sample.get("_true_label", 0)

    return {
        "sample_index": idx,
        "features": raw_values,
        "true_label": true_label,
        "ml": ml_result,
    }


@app.get("/api/globe_data")
async def globe_data(session_id: str = Query(...)):
    """
    Return PCA 3D coordinates for all test samples.
    PCA(n_components=3) on scaled test features.
    """
    _require_session(session_id)
    cfg = _require_column_config(session_id)
    test_samples = load_json(session_id, "test_samples.json")

    if not test_samples:
        raise HTTPException(status_code=404, detail="No test samples available")

    feature_cols = cfg.get("feature_cols", [])
    n_samples = len(test_samples)

    # Build feature matrix
    X = np.array([
        [s.get(f, 0.0) for f in feature_cols]
        for s in test_samples
    ])

    # Pad to >= 3 features if needed
    n_features = X.shape[1]
    if n_features < 3:
        pad = np.zeros((n_samples, 3 - n_features))
        X = np.hstack([X, pad])

    # Handle degenerate row count (< 3 rows)
    if n_samples < 3:
        while len(X) < 3:
            X = np.vstack([X, X[-1:]])
        test_samples = test_samples + [test_samples[-1]] * (3 - n_samples)

    # PCA
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=42)
    coords = pca.fit_transform(X)

    # Normalize: (coords / axis_std) × 4.5
    axis_stds = coords.std(axis=0)
    axis_stds[axis_stds == 0] = 1.0
    coords_norm = (coords / axis_stds) * 4.5

    # Get anomaly scores from best model
    from pipeline import load_model
    metrics = load_json(session_id, "metrics.json") or {}
    best_model_name = "logistic_regression"
    best_auc = 0
    for mn, mv in metrics.items():
        if mv.get("auc", 0) > best_auc:
            best_auc = mv["auc"]
            best_model_name = mn

    best_model = load_model(session_id, best_model_name)

    # Compute anomaly scores
    if best_model is not None:
        try:
            scores = best_model.predict_proba(X[:len(test_samples)])[:, 1]
        except Exception:
            scores = np.zeros(len(test_samples))
    else:
        scores = np.zeros(len(test_samples))

    # Get top features by absolute PCA components
    pca_components = np.abs(pca.components_)  # shape (3, n_features)

    globe_points = []
    for i, sample in enumerate(test_samples[:n_samples]):
        # Top features for this point (highest contributing features)
        top_feat_indices = np.argsort(-pca_components.sum(axis=0))[:3]
        top_features = [
            {"feature": feature_cols[j] if j < len(feature_cols) else f"f{j}",
             "importance": float(pca_components[:, j].sum())}
            for j in top_feat_indices
        ]

        globe_points.append({
            "id": i,
            "x": float(coords_norm[i, 0]),
            "y": float(coords_norm[i, 1]),
            "z": float(coords_norm[i, 2]),
            "true_label": int(sample.get("_true_label", 0)),
            "predicted_label": int(float(scores[i]) >= 0.5),
            "anomaly_score": float(scores[i]),
            "top_features": top_features,
        })

    return {
        "points": globe_points,
        "variance_explained": [float(v) for v in pca.explained_variance_ratio_],
        "total_variance": float(pca.explained_variance_ratio_.sum()),
        "n_points": len(globe_points),
    }


@app.post("/api/inject_point")
async def inject_point(req: InjectPointRequest):
    """
    Score an injected point and return its 3D PCA coordinates + verdict.
    """
    _require_session(req.session_id)
    cfg = _require_column_config(req.session_id)
    test_samples = load_json(req.session_id, "test_samples.json")

    feature_cols = cfg.get("feature_cols", [])
    X_scaled, _ = _scale_features(req.features, req.session_id, cfg)

    from pipeline import predict_all_models
    ml_result = predict_all_models(X_scaled, req.session_id)

    # Compute PCA position relative to training distribution
    if test_samples:
        X_base = np.array([
            [s.get(f, 0.0) for f in feature_cols]
            for s in test_samples
        ])
    else:
        X_base = X_scaled

    if X_base.shape[0] < 3:
        X_base = np.vstack([X_base] * (3 - X_base.shape[0] + 1))

    n_features = X_base.shape[1]
    if n_features < 3:
        X_base = np.hstack([X_base, np.zeros((X_base.shape[0], 3 - n_features))])

    from sklearn.decomposition import PCA
    pca = PCA(n_components=3, random_state=42)
    pca.fit(X_base)

    # Project injected point
    inject_raw = np.array([req.features.get(f, 0.0) for f in feature_cols]).reshape(1, -1)
    if inject_raw.shape[1] < 3:
        inject_raw = np.hstack([inject_raw, np.zeros((1, 3 - inject_raw.shape[1]))])

    coords_base = pca.transform(X_base)
    axis_stds = coords_base.std(axis=0)
    axis_stds[axis_stds == 0] = 1.0

    inject_scaled_for_pca = inject_raw
    coord = pca.transform(inject_scaled_for_pca)
    coord_norm = (coord / axis_stds) * 4.5

    return {
        "x": float(coord_norm[0, 0]),
        "y": float(coord_norm[0, 1]),
        "z": float(coord_norm[0, 2]),
        "verdict": ml_result["verdict"],
        "anomaly_score": ml_result["ensemble_score"],
        "flags": ml_result["flags"],
        "probabilities": ml_result["probabilities"],
    }


@app.get("/api/leaderboard")
async def leaderboard(session_id: str = Query(...)):
    """Return top 10 riskiest test transactions by anomaly score."""
    _require_session(session_id)
    test_samples = load_json(session_id, "test_samples.json")
    cfg = _require_column_config(session_id)

    if not test_samples:
        return {"leaderboard": []}

    feature_cols = cfg.get("feature_cols", [])

    # Score all samples with best model
    from pipeline import load_model
    metrics = load_json(session_id, "metrics.json") or {}
    best_model_name = "logistic_regression"
    best_auc = 0
    for mn, mv in metrics.items():
        if mv.get("auc", 0) > best_auc:
            best_auc = mv["auc"]
            best_model_name = mn

    best_model = load_model(session_id, best_model_name)

    if best_model is None:
        return {"leaderboard": []}

    X = np.array([[s.get(f, 0.0) for f in feature_cols] for s in test_samples])

    try:
        scores = best_model.predict_proba(X)[:, 1]
    except Exception:
        scores = np.zeros(len(test_samples))

    # Determine verdict from flags
    from pipeline import predict_all_models

    # Build leaderboard entries
    entries = []
    for i, (sample, score) in enumerate(zip(test_samples, scores)):
        verdict = "HIGH_RISK" if score >= 0.7 else "LIKELY_ANOMALY" if score >= 0.5 else "LIKELY_NORMAL"
        entries.append({
            "rank": 0,
            "index": i,
            "anomaly_score": float(score),
            "true_label": int(sample.get("_true_label", 0)),
            "verdict": verdict,
            "features": sample.get("_raw_values", {}),
        })

    # Sort by score descending, take top 10
    entries.sort(key=lambda x: -x["anomaly_score"])
    top10 = entries[:10]
    for i, entry in enumerate(top10):
        entry["rank"] = i + 1

    return {"leaderboard": top10}


@app.get("/api/case_list")
async def case_list(session_id: str = Query(...)):
    """Return saved investigation cases for a session."""
    _require_session(session_id)
    cases = load_json(session_id, "cases.json") or []
    return {"cases": cases}


@app.post("/api/case_save")
async def case_save(req: CaseSaveRequest):
    """Save an investigation decision (flag/allow/escalate/note)."""
    _require_session(req.session_id)

    cases = load_json(req.session_id, "cases.json") or []

    new_case = {
        "id": str(uuid.uuid4()),
        "transaction_id": req.transaction_id,
        "decision": req.decision,
        "note": req.note,
        "transaction_data": req.transaction_data,
        "risk_score": req.risk_score,
        "verdict": req.verdict,
        "timestamp": time.time(),
        "timestamp_iso": pd.Timestamp.now().isoformat(),
    }
    cases.insert(0, new_case)

    # Keep last 100 cases
    cases = cases[:100]
    save_json(req.session_id, "cases.json", cases)

    return {"success": True, "case_id": new_case["id"]}


@app.post("/api/test_dataset")
async def test_dataset(
    file: UploadFile = File(...),
    session_id: str = Form(...),
):
    """
    Upload an unlabelled CSV and run inference using the session's best-trained model.
    Returns: { "normal": [...], "fraud": [...], "total": int, "feature_cols": [...] }
    Each item contains all feature values + predicted_label + probability.
    """
    _require_session(session_id)
    cfg = _require_column_config(session_id)
    _require_metrics(session_id)

    # ── Read uploaded CSV ─────────────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Only .csv files are accepted")

    try:
        content = await file.read()
        if not content:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        df = pd.read_csv(io.BytesIO(content), low_memory=False)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV file: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=400, detail="CSV file contains no rows")

    feature_cols: list[str] = cfg["feature_cols"]
    original_cols = cfg.get("original_feature_cols", feature_cols)

    # ── Scale features ────────────────────────────────────────────────────────
    from pipeline import load_scaler
    scaler = load_scaler(session_id)
    if scaler is None:
        raise HTTPException(status_code=404, detail="Scaler not found — retrain model")

    if hasattr(scaler, "transformers_"):
        col_type_map = load_json(session_id, "col_type_map.json") or {}
        # Ensure we construct a dataframe with exact columns in correct order
        transform_cols = list(col_type_map.keys()) if col_type_map else original_cols
        
        df_transform = pd.DataFrame(index=df.index)
        for col in transform_cols:
            if col in df.columns:
                df_transform[col] = df[col]
            else:
                df_transform[col] = None

        for col, ctype in col_type_map.items():
            if ctype in ("low_card_cat", "high_card_cat") and col in df_transform.columns:
                df_transform[col] = df_transform[col].astype(str).replace(["None", "nan", "NaN"], np.nan)
        try:
            X_scaled = scaler.transform(df_transform)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"ColumnTransformer failed: {str(e)}")
    else:
        for col in original_cols:
            if col not in df.columns:
                df[col] = 0.0
        X_raw = df[original_cols].fillna(0.0).values.astype(float)
        X_scaled = scaler.transform(X_raw)

    # ── Load best model ───────────────────────────────────────────────────────
    from pipeline import load_model
    metrics = load_json(session_id, "metrics.json") or {}
    best_model_name = "logistic_regression"
    best_auc = 0.0
    for mn, mv in metrics.items():
        if mv.get("auc", 0) > best_auc:
            best_auc = mv["auc"]
            best_model_name = mn

    best_model = load_model(session_id, best_model_name)
    if best_model is None:
        raise HTTPException(status_code=404, detail="No trained model found — retrain model")

    # ── Run inference ─────────────────────────────────────────────────────────
    try:
        probabilities = best_model.predict_proba(X_scaled)[:, 1]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {str(e)}")

    # ── Build results ─────────────────────────────────────────────────────────
    normal_list = []
    fraud_list = []

    for i, prob in enumerate(probabilities):
        predicted_label = 1 if prob >= 0.5 else 0
        # Use original df values for display (before scaling)
        display_values = {}
        for col in df.columns:
            val = df.iloc[i][col]
            try:
                display_values[col] = float(val) if pd.notna(val) else None
            except (ValueError, TypeError):
                display_values[col] = str(val) if pd.notna(val) else None

        item = {
            "index": i,
            "transaction_id": f"TXN-{i+1:05d}",
            "predicted_label": predicted_label,
            "probability": round(float(prob), 4),
            "confidence": round(float(max(prob, 1 - prob)), 4),
            "verdict": "FRAUD" if predicted_label == 1 else "NORMAL",
            "features": display_values,
        }

        if predicted_label == 1:
            fraud_list.append(item)
        else:
            normal_list.append(item)

    # Sort fraud by probability descending (highest risk first)
    fraud_list.sort(key=lambda x: -x["probability"])

    return {
        "normal": normal_list,
        "fraud": fraud_list,
        "total": len(normal_list) + len(fraud_list),
        "fraud_count": len(fraud_list),
        "normal_count": len(normal_list),
        "feature_cols": feature_cols,
        "model_used": best_model_name,
    }


@app.get("/api/export_csv")
async def export_csv(session_id: str = Query(...)):
    """Download all scored test samples as CSV."""
    _require_session(session_id)
    test_samples = load_json(session_id, "test_samples.json")
    cfg = load_json(session_id, "column_config.json")

    if not test_samples or not cfg:
        raise HTTPException(status_code=404, detail="No data available")

    feature_cols = cfg.get("feature_cols", [])

    from pipeline import load_model
    metrics = load_json(session_id, "metrics.json") or {}
    best_model_name = "logistic_regression"
    best_auc = 0
    for mn, mv in metrics.items():
        if mv.get("auc", 0) > best_auc:
            best_auc = mv["auc"]
            best_model_name = mn
    best_model = load_model(session_id, best_model_name)

    X = np.array([[s.get(f, 0.0) for f in feature_cols] for s in test_samples])
    if best_model is not None:
        try:
            scores = best_model.predict_proba(X)[:, 1]
        except Exception:
            scores = np.zeros(len(test_samples))
    else:
        scores = np.zeros(len(test_samples))

    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = feature_cols + ["true_label", "anomaly_score", "verdict"]
    writer.writerow(header)

    # Rows
    for i, sample in enumerate(test_samples):
        raw = sample.get("_raw_values", {})
        row = [raw.get(f, "") for f in feature_cols]
        score = float(scores[i])
        verdict = "HIGH_RISK" if score >= 0.7 else "LIKELY_ANOMALY" if score >= 0.5 else "LIKELY_NORMAL"
        row += [sample.get("_true_label", 0), round(score, 4), verdict]
        writer.writerow(row)

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="anomalyiq_export_{session_id[:8]}.csv"'}
    )


class BlacklistRequest(BaseModel):
    session_id: str
    accounts: list[dict]


@app.get("/api/fraud_rings")
async def get_fraud_rings(session_id: str = Query(...)):
    """Return Louvain community fraud rings and LLM narratives."""
    _require_session(session_id)
    graph_result = load_json(session_id, "graph_analysis.json")
    if not graph_result or "fraud_rings" not in graph_result:
        return {"rings": []}
    
    rings = graph_result.get("fraud_rings", [])
    
    # Generate LLM summaries lazily — only ONE per request to respect
    # the free-tier rate limit of 8 req/min on OpenRouter.
    # The frontend polls this endpoint, so subsequent calls will fill in the rest.
    from explainer import generate_cluster_narrative
    ctx = load_session_context(session_id)
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    
    updated = False
    for ring in rings[:10]:
        if ring.get("summary") in ("Pending AI analysis...", None, ""):
            try:
                ring["summary"] = generate_cluster_narrative(ring, ctx, api_key=api_key)
            except Exception as e:
                # If rate-limited, leave as pending so next poll will retry
                if "429" in str(e):
                    ring["summary"] = "Pending AI analysis..."
                else:
                    ring["summary"] = "AI analysis unavailable."
            updated = True
            break  # ← Only process ONE pending ring per request
            
    if updated:
        save_json(session_id, "graph_analysis.json", graph_result)
        
    return {"rings": rings}


@app.post("/api/blacklist")
async def add_to_blacklist(req: BlacklistRequest):
    """Add accounts from a fraud ring to the organizational blacklist."""
    _require_session(req.session_id)
    blacklist = load_json(req.session_id, "blacklist.json") or []
    
    existing_ccs = {b.get("cc_num") for b in blacklist if "cc_num" in b}
    
    added_count = 0
    import datetime
    for acc in req.accounts:
        cc_num = acc.get("cc_num")
        if cc_num and cc_num not in existing_ccs:
            acc["added_at"] = datetime.datetime.utcnow().isoformat() + "Z"
            blacklist.append(acc)
            existing_ccs.add(cc_num)
            added_count += 1
            
    save_json(req.session_id, "blacklist.json", blacklist)
    return {"message": f"Added {added_count} accounts to blacklist", "total": len(blacklist)}


@app.get("/api/blacklist")
async def get_blacklist(session_id: str = Query(...)):
    """Retrieve the organizational blacklist."""
    _require_session(session_id)
    blacklist = load_json(session_id, "blacklist.json") or []
    return {"blacklist": blacklist}
