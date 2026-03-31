"""
pipeline.py — AnomalyIQ ML Preprocessing + Training + Evaluation

preprocess_data()     — Null drop, encode target, StandardScaler, SMOTE, save artifacts
train_*()             — Train each of the 4 model types
evaluate_model()      — Compute full metrics dict per model
run_full_pipeline()   — Orchestrate all steps, 9-step status callbacks
"""

import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable, Optional, Any

# Try TensorFlow import
TF_AVAILABLE = False
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    pass

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve
)
import joblib

from session_manager import (
    ensure_session_dir, save_json, get_session_file_path, save_training_status
)
from preprocessor import fit_preprocessor, load_preprocessor


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_data(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    session_id: str,
) -> dict:
    """
    Full preprocessing pipeline via ColumnTransformer.

    Steps:
      1. Call fit_preprocessor() — handles all column types automatically:
         numeric → StandardScaler
         datetime → DatetimeFeatureExtractor (hour, day_of_week, age_years)
         low_card_cat → OneHotEncoder
         high_card_cat → TargetEncoder
      2. Compute col_stats on numeric/datetime output columns only
      3. Compute class_means per post-transform numeric feature
      4. Stratified 80/20 train-test split
      5. Apply SMOTE to training data only (when minority/majority < 0.1)
      6. Save test_samples.json (includes _raw_categorical_values for Graph Engine)
      7. Save column_config.json

    Returns dict with X_train, X_test, y_train, y_test, col_stats, etc.
    """
    session_dir = ensure_session_dir(session_id)

    # ── Step 1: Fit ColumnTransformer ────────────────────────────────────────
    prep = fit_preprocessor(df, feature_cols, target_col, session_id)

    X = prep["X_transformed"]          # float64 numpy array, fully transformed
    y = prep["y"]                      # 0/1 encoded target
    feature_names_out = prep["feature_names_out"]
    raw_df_features = prep["raw_df_features"]   # original strings for Graph Engine
    col_type_map = prep["col_type_map"]
    n_numeric_out = prep["n_numeric_out"]

    # ── Step 2: Compute col_stats (numeric output cols only) ──────────────────
    # Only meaningful to compute stats for scaled numeric features, not OHE bits
    col_stats = {}
    for i, feat in enumerate(feature_names_out[:n_numeric_out]):
        col_data = X[:, i]
        col_stats[feat] = {
            "min":  float(np.min(col_data)),
            "max":  float(np.max(col_data)),
            "mean": float(np.mean(col_data)),
            "std":  float(np.std(col_data)) or 1.0,
        }

    # ── Step 3: Class means ───────────────────────────────────────────────────
    class_means = {0: {}, 1: {}}
    for cls in [0, 1]:
        mask = (y == cls)
        if mask.sum() > 0:
            X_cls = X[mask]
            for i, feat in enumerate(feature_names_out[:n_numeric_out]):
                class_means[cls][feat] = float(np.mean(X_cls[:, i]))
        else:
            for feat in feature_names_out[:n_numeric_out]:
                class_means[cls][feat] = col_stats.get(feat, {}).get("mean", 0.0)

    # ── Step 4: Stratified 80/20 split ───────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # Also split the raw (pre-encoded) data for Graph Engine
    raw_arr = raw_df_features.values
    raw_cols = list(raw_df_features.columns)

    _, raw_test = train_test_split(
        raw_arr,
        test_size=0.20,
        stratify=y,
        random_state=42
    )

    # ── Step 5: Conditional SMOTE ────────────────────────────────────────────
    smote_applied = False
    minority_count = np.sum(y_train == 1)
    majority_count = np.sum(y_train == 0)
    ratio = minority_count / max(majority_count, 1)

    if ratio < 0.1 and minority_count >= 6:
        try:
            from imblearn.over_sampling import SMOTE
            target_strategy = min(0.5, ratio * 5)
            smote = SMOTE(
                sampling_strategy=target_strategy,
                random_state=42,
                k_neighbors=min(5, minority_count - 1)
            )
            X_train, y_train = smote.fit_resample(X_train, y_train)
            smote_applied = True
        except Exception:
            pass  # SMOTE failure is non-fatal

    # ── Step 6: Save test samples ─────────────────────────────────────────────
    n_test = min(5000, len(X_test))
    test_samples = []
    for idx in range(n_test):
        # Scaled (transformed) feature values  — fed to ML models
        sample = {feat: float(X_test[idx, j]) for j, feat in enumerate(feature_names_out)}

        # Raw numeric values for interpretability (take numeric cols only)
        raw_numeric = {}
        for j, feat in enumerate(feature_names_out[:n_numeric_out]):
            raw_numeric[feat] = float(X_test[idx, j])
        sample["_raw_values"] = raw_numeric

        # Raw categorical values for Graph Engine — the critical bridge
        raw_categ = {}
        for j, col in enumerate(raw_cols):
            if col_type_map.get(col) in ("low_card_cat", "high_card_cat"):
                raw_categ[col] = str(raw_test[idx, j])
        sample["_raw_categorical_values"] = raw_categ

        sample["_true_label"] = int(y_test[idx])
        test_samples.append(sample)

    save_json(session_id, "test_samples.json", test_samples)

    # ── Save column_config ────────────────────────────────────────────────────
    column_config = {
        "feature_cols": feature_names_out,        # post-transform names
        "original_feature_cols": feature_cols,    # original names (for UI display)
        "col_type_map": col_type_map,
        "target_col": target_col,
        "n_features": len(feature_names_out),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_total": int(len(X)),
        "col_stats": col_stats,
        "class_means": class_means,
        "smote_applied": smote_applied,
        "tf_available": TF_AVAILABLE,
        "class_distribution": {
            "class_0": int(np.sum(y == 0)),
            "class_1": int(np.sum(y == 1)),
        }
    }
    save_json(session_id, "column_config.json", column_config)

    return {
        "X_train":       X_train,
        "X_test":        X_test,
        "y_train":       y_train,
        "y_test":        y_test,
        "col_stats":     col_stats,
        "class_means":   class_means,
        "feature_cols":  feature_names_out,
        "smote_applied": smote_applied,
        "n_features":    len(feature_names_out),
    }


def _encode_target(y_raw: pd.Series) -> tuple[np.ndarray, int]:
    """
    Encode target column to 0/1 integers.

    Detects binary strings ('0','1','true','false','yes','no').
    Ensures minority class = 1. Returns (encoded_array, minority_class_value).
    """
    # Check if already numeric 0/1
    try:
        y_numeric = pd.to_numeric(y_raw)
        unique_vals = set(y_numeric.unique())
        if unique_vals <= {0, 1}:
            return y_numeric.values.astype(int), 1
    except (TypeError, ValueError):
        pass

    # Check for string boolean representations
    y_str = y_raw.astype(str).str.lower().str.strip()
    str_vals = set(y_str.unique())

    bool_maps = [
        ({"true", "false"}, {"true": 1, "false": 0}),
        ({"yes", "no"}, {"yes": 1, "no": 0}),
        ({"1", "0"}, {"1": 1, "0": 0}),
        ({"1.0", "0.0"}, {"1.0": 1, "0.0": 0}),
    ]

    for known_set, mapping in bool_maps:
        if str_vals <= known_set:
            y_encoded = y_str.map(mapping).fillna(0).astype(int)
            return _ensure_minority_is_1(y_encoded.values), 1

    # LabelEncoder fallback
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_raw.values)
    return _ensure_minority_is_1(y_encoded), 1


def _ensure_minority_is_1(y: np.ndarray) -> np.ndarray:
    """Flip 0/1 if the current '1' class is not the minority class."""
    count_1 = np.sum(y == 1)
    count_0 = np.sum(y == 0)
    if count_1 > count_0:
        # Minority is labeled 0 — flip
        return 1 - y
    return y


# ── Model Training ────────────────────────────────────────────────────────────

def train_logistic_regression(X_train: np.ndarray, y_train: np.ndarray, session_id: str) -> Any:
    """Train Logistic Regression: balanced class_weight, max_iter=1000."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, get_session_file_path(session_id, "lr_model.pkl"))
    return model


def train_svm(X_train: np.ndarray, y_train: np.ndarray, session_id: str) -> Any:
    """Train SVM: RBF kernel, probability=True, cap at 8000 rows."""
    if len(X_train) > 8000:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(X_train), 8000, replace=False)
        X_sub = X_train[idx]
        y_sub = y_train[idx]
    else:
        X_sub, y_sub = X_train, y_train

    model = SVC(
        kernel="rbf",
        probability=True,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_sub, y_sub)
    joblib.dump(model, get_session_file_path(session_id, "svm_model.pkl"))
    return model


def train_decision_tree(X_train: np.ndarray, y_train: np.ndarray, session_id: str) -> Any:
    """
    Train Decision Tree.
    If ≤20K rows: CV over depths 3–10 to find best depth.
    Otherwise: fixed depth=6.
    """
    from sklearn.model_selection import cross_val_score

    if len(X_train) <= 20000:
        best_depth = 6
        best_score = 0.0
        for depth in range(3, 11):
            dt = DecisionTreeClassifier(
                max_depth=depth,
                class_weight="balanced",
                random_state=42,
            )
            scores = cross_val_score(dt, X_train, y_train, cv=3, scoring="roc_auc", n_jobs=-1)
            mean_score = scores.mean()
            if mean_score > best_score:
                best_score = mean_score
                best_depth = depth
    else:
        best_depth = 6

    model = DecisionTreeClassifier(
        max_depth=best_depth,
        class_weight="balanced",
        random_state=42,
    )
    model.fit(X_train, y_train)
    model.best_depth_ = best_depth
    joblib.dump(model, get_session_file_path(session_id, "dt_model.pkl"))
    return model


def train_neural_network(X_train: np.ndarray, y_train: np.ndarray, session_id: str) -> Any:
    """
    Train Neural Network.
    Uses Keras (128→Dropout→64→Dropout→32→1 sigmoid) if TF available.
    Falls back to MLPClassifier(128,64,32).
    """
    if TF_AVAILABLE:
        model = _train_keras_nn(X_train, y_train, session_id)
    else:
        model = _train_mlp(X_train, y_train, session_id)
    return model


def _train_keras_nn(X_train: np.ndarray, y_train: np.ndarray, session_id: str) -> Any:
    """Train Keras neural network: 128→Dropout→64→Dropout→32→1 sigmoid."""
    import tensorflow as tf
    from tensorflow import keras

    n_features = X_train.shape[1]

    # Compute class weights for imbalanced data
    n_total = len(y_train)
    n_pos = np.sum(y_train == 1)
    n_neg = n_total - n_pos
    class_weight_0 = n_total / (2 * max(n_neg, 1))
    class_weight_1 = n_total / (2 * max(n_pos, 1))

    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.fit(
        X_train, y_train,
        epochs=30,
        batch_size=64,
        class_weight={0: class_weight_0, 1: class_weight_1},
        validation_split=0.1,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)
        ]
    )

    model_path = get_session_file_path(session_id, "nn_model.keras")
    model.save(str(model_path))

    # Wrap in a sklearn-compatible wrapper
    return KerasWrapper(model, model_path)


class KerasWrapper:
    """sklearn-compatible wrapper around a Keras model for uniform API."""

    def __init__(self, model: Any, model_path: Path):
        self.model = model
        self.model_path = model_path
        self._name = "keras_nn"

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probs = self.model.predict(X, verbose=0).flatten()
        return np.column_stack([1 - probs, probs])

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


def _train_mlp(X_train: np.ndarray, y_train: np.ndarray, session_id: str) -> Any:
    """Train MLPClassifier as Keras fallback."""
    model = MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    model.fit(X_train, y_train)
    joblib.dump(model, get_session_file_path(session_id, "nn_model.pkl"))
    return model


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    model_name: str,
) -> dict:
    """
    Evaluate a trained model on the test set.

    Returns:
        {accuracy, precision, recall, f1, auc, inference_time_ms,
         confusion_matrix: {tn, fp, fn, tp},
         roc_curve: {fpr[], tpr[]},
         pr_curve: {precision[], recall[]},
         feature_importances (DT only),
         coefficients (LR only)}
    """
    t0 = time.perf_counter()
    y_proba = model.predict_proba(X_test)[:, 1]
    inference_ms = (time.perf_counter() - t0) * 1000

    y_pred = (y_proba >= 0.5).astype(int)

    # Basic metrics
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))

    # AUC
    try:
        auc = float(roc_auc_score(y_test, y_proba))
    except Exception:
        auc = 0.5

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)

    # ROC curve (downsample for payload size)
    try:
        fpr_arr, tpr_arr, _ = roc_curve(y_test, y_proba)
        fpr_list, tpr_list = _downsample_curve(fpr_arr.tolist(), tpr_arr.tolist())
    except Exception:
        fpr_list, tpr_list = [0.0, 1.0], [0.0, 1.0]

    # PR curve
    try:
        prec_arr, rec_arr, _ = precision_recall_curve(y_test, y_proba)
        prec_list, rec_list = _downsample_curve(prec_arr.tolist(), rec_arr.tolist())
    except Exception:
        prec_list, rec_list = [1.0, 0.0], [0.0, 1.0]

    result = {
        "model_name": model_name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "inference_time_ms": round(inference_ms, 2),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "roc_curve": {"fpr": fpr_list, "tpr": tpr_list},
        "pr_curve": {"precision": prec_list, "recall": rec_list},
    }

    # Feature importances for Decision Tree
    if model_name == "decision_tree" and hasattr(model, "feature_importances_"):
        fi = dict(zip(feature_cols, model.feature_importances_.tolist()))
        result["feature_importances"] = {
            k: round(v, 4) for k, v in
            sorted(fi.items(), key=lambda x: -x[1])
        }

    # Coefficients for Logistic Regression (absolute values)
    if model_name == "logistic_regression" and hasattr(model, "coef_"):
        coef = np.abs(model.coef_[0])
        coef_dict = dict(zip(feature_cols, coef.tolist()))
        result["coefficients"] = {
            k: round(v, 4) for k, v in
            sorted(coef_dict.items(), key=lambda x: -x[1])
        }

    return result


def _downsample_curve(xs: list, ys: list, max_points: int = 200) -> tuple[list, list]:
    """Downsample ROC/PR curve arrays to reduce JSON payload size."""
    if len(xs) <= max_points:
        return xs, ys
    step = len(xs) // max_points
    xs_ds = xs[::step] + [xs[-1]]
    ys_ds = ys[::step] + [ys[-1]]
    return xs_ds, ys_ds


# ── Full Pipeline ─────────────────────────────────────────────────────────────

def run_full_pipeline(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    session_id: str,
    status_callback: Optional[Callable[[int, str], None]] = None,
) -> dict:
    """
    Full training pipeline with 9 status step callbacks.

    Steps:
      1 = Uploading    2 = Preprocessing   3 = SMOTE
      4 = LR           5 = SVM             6 = DT
      7 = NN           8 = Evaluating      9 = Ready

    Returns: metrics dict with all 4 models.
    """

    def step(n: int, msg: str):
        save_training_status(session_id, n, msg, done=False)
        if status_callback:
            status_callback(n, msg)

    step(1, "Uploading and validating dataset...")
    time.sleep(0.2)  # Brief pause for UI to register

    # ── Preprocessing ────────────────────────────────────────────────────────
    step(2, "Preprocessing data — encoding, scaling, splitting...")
    prep = preprocess_data(df, feature_cols, target_col, session_id)

    X_train = prep["X_train"]
    X_test = prep["X_test"]
    y_train = prep["y_train"]
    y_test = prep["y_test"]
    feature_cols = prep["feature_cols"]

    # ── SMOTE status ─────────────────────────────────────────────────────────
    if prep["smote_applied"]:
        step(3, "Applying SMOTE to balance class distribution...")
    else:
        step(3, "Class balance acceptable — SMOTE not required")
    time.sleep(0.1)

    metrics = {}
    train_times = {}

    # ── Train Logistic Regression ─────────────────────────────────────────────
    step(4, "Training Logistic Regression model...")
    t0 = time.perf_counter()
    lr_model = train_logistic_regression(X_train, y_train, session_id)
    train_times["logistic_regression"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Train SVM ─────────────────────────────────────────────────────────────
    step(5, "Training Support Vector Machine (RBF kernel)...")
    t0 = time.perf_counter()
    svm_model = train_svm(X_train, y_train, session_id)
    train_times["svm"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Train Decision Tree ───────────────────────────────────────────────────
    step(6, "Training Decision Tree with cross-validated depth search...")
    t0 = time.perf_counter()
    dt_model = train_decision_tree(X_train, y_train, session_id)
    train_times["decision_tree"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Train Neural Network ──────────────────────────────────────────────────
    step(7, "Training Neural Network (MLP 128→64→32)...")
    t0 = time.perf_counter()
    nn_model = train_neural_network(X_train, y_train, session_id)
    train_times["neural_network"] = round((time.perf_counter() - t0) * 1000, 1)

    # ── Evaluate all models ───────────────────────────────────────────────────
    step(8, "Evaluating all 4 models on held-out test set...")

    model_map = {
        "logistic_regression": lr_model,
        "svm": svm_model,
        "decision_tree": dt_model,
        "neural_network": nn_model,
    }

    for model_name, model in model_map.items():
        eval_result = evaluate_model(model, X_test, y_test, feature_cols, model_name)
        eval_result["train_time_ms"] = train_times.get(model_name, 0.0)
        metrics[model_name] = eval_result

    # Save metrics
    save_json(session_id, "metrics.json", metrics)

    # ── Mark complete ─────────────────────────────────────────────────────────
    step(9, "✓ All models trained and ready")
    save_training_status(session_id, 9, "All models trained and ready", done=True)

    return metrics


def load_model(session_id: str, model_name: str) -> Any:
    """Load a trained model from session directory."""
    if model_name == "neural_network" and TF_AVAILABLE:
        keras_path = get_session_file_path(session_id, "nn_model.keras")
        if keras_path.exists():
            import tensorflow as tf
            keras_model = tf.keras.models.load_model(str(keras_path))
            return KerasWrapper(keras_model, keras_path)

    # Fallback to joblib
    pkl_map = {
        "logistic_regression": "lr_model.pkl",
        "svm": "svm_model.pkl",
        "decision_tree": "dt_model.pkl",
        "neural_network": "nn_model.pkl",
    }
    pkl_name = pkl_map.get(model_name)
    if pkl_name:
        pkl_path = get_session_file_path(session_id, pkl_name)
        if pkl_path.exists():
            return joblib.load(pkl_path)

    return None


def load_scaler(session_id: str):
    """
    Backwards-compatible loader: tries to load the new preprocessor.pkl first,
    then falls back to the legacy scaler.pkl for sessions trained before
    the ColumnTransformer upgrade.
    """
    # Try new-style preprocessor first
    ct = load_preprocessor(session_id)
    if ct is not None:
        return ct

    # Legacy fallback
    scaler_path = get_session_file_path(session_id, "scaler.pkl")
    if scaler_path.exists():
        return joblib.load(scaler_path)
    return None


def predict_all_models(
    X_scaled: np.ndarray,
    session_id: str,
) -> dict:
    """
    Run prediction on all 4 models for a single sample.
    Returns probabilities, predictions, flags, verdict, disagreement_index.
    """
    model_names = ["logistic_regression", "svm", "decision_tree", "neural_network"]
    probabilities = {}
    predictions = {}

    for name in model_names:
        model = load_model(session_id, name)
        if model is not None:
            try:
                prob = float(model.predict_proba(X_scaled)[:, 1][0])
                pred = int(prob >= 0.5)
            except Exception:
                prob = 0.0
                pred = 0
        else:
            prob = None
            pred = 0
        probabilities[name] = prob
        predictions[name] = pred

    valid_probs = [p for p in probabilities.values() if p is not None]
    flags = sum(1 for p in valid_probs if p is not None and p >= 0.5)

    # Disagreement index: std of valid probabilities × 200, capped at 100
    if len(valid_probs) >= 2:
        std = float(np.std(valid_probs))
        disagreement_index = min(100.0, std * 200)
    else:
        disagreement_index = 0.0

    # Verdict
    if flags == 4:
        verdict = "HIGH_RISK"
    elif flags == 3:
        verdict = "LIKELY_ANOMALY"
    elif flags == 2:
        verdict = "AMBIGUOUS"
    else:
        verdict = "LIKELY_NORMAL"

    # Ensemble score (mean of valid probs)
    ensemble_score = float(np.mean(valid_probs)) if valid_probs else 0.0

    return {
        "probabilities": probabilities,
        "predictions": predictions,
        "flags": flags,
        "verdict": verdict,
        "disagreement_index": round(disagreement_index, 2),
        "ensemble_score": round(ensemble_score, 4),
    }
