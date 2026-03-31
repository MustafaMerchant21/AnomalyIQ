"""
preprocessor.py — AnomalyIQ ColumnTransformer-Based Preprocessing

Handles all column types automatically:
  - numeric          → SimpleImputer(median) → StandardScaler
  - datetime         → DatetimeFeatureExtractor (extracts hour, day_of_week, age_years)
  - low_card_cat     → SimpleImputer(most_frequent) → OneHotEncoder (≤ 20 unique values)
  - high_card_cat    → SimpleImputer(most_frequent) → TargetEncoder  (> 20 unique values)

Public API:
  detect_column_types(df, feature_cols)                 → col_type_map
  fit_preprocessor(df, feature_cols, target_col, sid)   → dict of transformed data
  transform_single_row(raw_features, session_id)        → (X_transformed, feature_names_out)
  load_preprocessor(session_id)                         → fitted ColumnTransformer
"""

from __future__ import annotations

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from typing import Any, Optional

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.preprocessing import TargetEncoder

from session_manager import (
    ensure_session_dir, save_json, get_session_file_path
)

# ── Constants ──────────────────────────────────────────────────────────────────

LOW_CARD_THRESHOLD = 20   # ≤ this unique count → OHE; > → TargetEncoder
DATETIME_KEYWORDS = [
    "date", "time", "timestamp", "dob", "birth", "created", "trans_date"
]


# ── Custom Transformer: Datetime Feature Extraction ───────────────────────────

class DatetimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Converts raw datetime strings into numeric features.

    Input : 1D array-like of datetime strings or Timestamps
    Output: 2D array with columns [hour_of_day, day_of_week, age_years]

    - hour_of_day  : 0–23 (from the datetime column)
    - day_of_week  : 0=Monday … 6=Sunday
    - age_years    : years since birth if column looks like a DOB, else 0.0

    Designed to be robust: any column that fails to parse just yields zeros.
    """

    def __init__(self, col_name: str = ""):
        self.col_name = col_name

    def fit(self, X, y=None):
        self.is_fitted_ = True
        return self

    def transform(self, X) -> np.ndarray:
        out = []
        # Convert to numpy array first, as X might be a pandas DataFrame
        for val in np.asarray(X).ravel():
            try:
                ts = pd.to_datetime(val, errors="coerce")
                if pd.isna(ts):
                    out.append([0.0, 0.0, 0.0])
                    continue

                hour = float(ts.hour)
                dow = float(ts.dayofweek)

                # Age: if column name contains "dob" or "birth"
                cn = self.col_name.lower()
                if any(k in cn for k in ("dob", "birth")):
                    age = float(
                        (pd.Timestamp.now() - ts).days / 365.25
                    )
                    age = max(0.0, min(age, 120.0))  # sanity cap
                else:
                    age = 0.0

                out.append([hour, dow, age])
            except Exception:
                out.append([0.0, 0.0, 0.0])

        return np.array(out, dtype=np.float64)

    def get_feature_names_out(self, input_features=None) -> list[str]:
        prefix = self.col_name or "dt"
        return [
            f"{prefix}_hour_of_day",
            f"{prefix}_day_of_week",
            f"{prefix}_age_years",
        ]


# ── Column Type Detection ──────────────────────────────────────────────────────

def detect_column_types(df: pd.DataFrame, feature_cols: list[str]) -> dict[str, str]:
    """
    Auto-classify each feature column into one of four roles:

      'numeric'          - castable to float64
      'datetime'         - parseable as a datetime string
      'low_card_cat'     - string with ≤ LOW_CARD_THRESHOLD unique values
      'high_card_cat'    - string with  > LOW_CARD_THRESHOLD unique values

    Returns:
        col_type_map: {col_name: role_string}
    """
    col_type_map: dict[str, str] = {}

    for col in feature_cols:
        if col not in df.columns:
            col_type_map[col] = "numeric"  # safe default
            continue

        series = df[col].dropna()

        if len(series) == 0:
            col_type_map[col] = "numeric"
            continue

        # 1. Try numeric first — fastest and most common
        try:
            pd.to_numeric(series, errors="raise")
            col_type_map[col] = "numeric"
            continue
        except (ValueError, TypeError):
            pass

        # 2. Check for datetime by name hint or content
        col_lower = col.lower()
        if any(kw in col_lower for kw in DATETIME_KEYWORDS):
            try:
                sample = series.iloc[:50]
                pd.to_datetime(sample, errors="raise")
                col_type_map[col] = "datetime"
                continue
            except Exception:
                pass

        # Also try datetime parse on content regardless of name
        try:
            sample = series.iloc[:50]
            parsed = pd.to_datetime(sample, errors="coerce")
            if parsed.notna().mean() >= 0.8:  # ≥80% successfully parsed
                col_type_map[col] = "datetime"
                continue
        except Exception:
            pass

        # 3. It's categorical — decide cardinality
        n_unique = series.nunique()
        if n_unique <= LOW_CARD_THRESHOLD:
            col_type_map[col] = "low_card_cat"
        else:
            col_type_map[col] = "high_card_cat"

    return col_type_map


# ── ColumnTransformer Builder ──────────────────────────────────────────────────

def build_column_transformer(col_type_map: dict[str, str]) -> ColumnTransformer:
    """
    Build a ColumnTransformer with one sub-pipeline per column type.

    Sub-pipelines:
      numeric       → median imputer → StandardScaler
      datetime      → DatetimeFeatureExtractor (per column, handles its own impute)
      low_card_cat  → most-frequent imputer → OHE (sparse_output=False, handle_unknown='ignore')
      high_card_cat → most-frequent imputer → TargetEncoder (binary target)

    Returns an UNFITTED ColumnTransformer ready for fit_transform().
    """
    numeric_cols     = [c for c, t in col_type_map.items() if t == "numeric"]
    datetime_cols    = [c for c, t in col_type_map.items() if t == "datetime"]
    low_card_cols    = [c for c, t in col_type_map.items() if t == "low_card_cat"]
    high_card_cols   = [c for c, t in col_type_map.items() if t == "high_card_cat"]

    transformers = []

    # ── Numeric pipeline ──────────────────────────────────────────────────────
    if numeric_cols:
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler",  StandardScaler()),
        ])
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    # ── Datetime columns (one pipeline per column for named output) ───────────
    for col in datetime_cols:
        dt_pipeline = Pipeline([
            ("extractor", DatetimeFeatureExtractor(col_name=col)),
        ])
        transformers.append((f"datetime_{col}", dt_pipeline, [col]))

    # ── Low-cardinality categoricals (OHE) ───────────────────────────────────
    if low_card_cols:
        low_card_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe",     OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
                dtype=np.float64,
            )),
        ])
        transformers.append(("low_card_cat", low_card_pipeline, low_card_cols))

    # ── High-cardinality categoricals (TargetEncoder) ─────────────────────────
    if high_card_cols:
        high_card_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", TargetEncoder(target_type="binary")),
        ])
        transformers.append(("high_card_cat", high_card_pipeline, high_card_cols))

    if not transformers:
        raise ValueError(
            "No valid columns to transform. "
            "Check that feature_cols exist in the dataframe."
        )

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",       # Drop any unrecognised columns cleanly
        verbose_feature_names_out=True,
    )
    return ct


# ── Feature Name Resolution ────────────────────────────────────────────────────

def _resolve_feature_names(
    ct: ColumnTransformer,
    col_type_map: dict[str, str],
) -> list[str]:
    """
    Extract the final list of output feature names from a fitted ColumnTransformer.
    Falls back to generic names f0, f1, ... if get_feature_names_out() fails.
    """
    try:
        return list(ct.get_feature_names_out())
    except Exception:
        # Fallback: count output columns from each transformer
        names = []
        for name, trans, cols in ct.transformers_:
            if name == "remainder":
                continue
            try:
                n = trans.transform(
                    pd.DataFrame([[None] * len(cols)], columns=cols)
                ).shape[1]
                for i in range(n):
                    names.append(f"{name}_{i}")
            except Exception:
                names.append(name)
        return names


# ── Public: Fit Preprocessor ───────────────────────────────────────────────────

def fit_preprocessor(
    df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    session_id: str,
) -> dict:
    """
    Detect column types, build + fit the ColumnTransformer, transform the data.

    Saves to session dir:
      preprocessor.pkl       — fitted ColumnTransformer pipeline
      col_type_map.json      — {col: role} classification
      feature_names_out.json — ordered list of post-transform feature names

    Returns:
        {
          X_transformed:       np.ndarray (n_samples, n_features_out)
          y:                   np.ndarray (n_samples,)  — encoded 0/1
          feature_names_out:   list[str]
          col_type_map:        dict[str, str]
          raw_df_features:     pd.DataFrame — original un-encoded columns for Graph Engine
          n_numeric_out:       int — count of numeric output features (for col_stats)
        }
    """
    session_dir = ensure_session_dir(session_id)

    # ── Target column ─────────────────────────────────────────────────────────
    cols_needed = feature_cols + [target_col]
    df = df[[c for c in cols_needed if c in df.columns]].dropna(
        subset=[target_col]
    ).copy()

    y_raw = df[target_col]
    from pipeline import _encode_target   # reuse existing robust encoder
    y, _ = _encode_target(y_raw)

    # Keep the raw feature DataFrame BEFORE any encoding
    # (passed to graph engine later as _raw_categorical_values)
    raw_df_features = df[feature_cols].copy()

    # ── Column type detection ─────────────────────────────────────────────────
    col_type_map = detect_column_types(df, feature_cols)

    # ── Build + fit ColumnTransformer ─────────────────────────────────────────
    ct = build_column_transformer(col_type_map)

    X_feat = df[feature_cols]

    # TargetEncoder needs y during fit
    X_transformed = ct.fit_transform(X_feat, y)

    # ── Feature names ─────────────────────────────────────────────────────────
    feature_names_out = _resolve_feature_names(ct, col_type_map)

    # ── Persist ───────────────────────────────────────────────────────────────
    preprocessor_path = session_dir / "preprocessor.pkl"
    joblib.dump(ct, preprocessor_path)

    save_json(session_id, "col_type_map.json", col_type_map)
    save_json(session_id, "feature_names_out.json", feature_names_out)

    # Count numeric output columns (for col_stats later)
    numeric_col_names = [c for c, t in col_type_map.items() if t == "numeric"]
    dt_col_names = [c for c, t in col_type_map.items() if t == "datetime"]
    # Numeric + datetime features come first in the output
    n_numeric_out = len(numeric_col_names) + len(dt_col_names) * 3

    return {
        "X_transformed":     X_transformed,
        "y":                 y,
        "feature_names_out": feature_names_out,
        "col_type_map":      col_type_map,
        "raw_df_features":   raw_df_features,
        "n_numeric_out":     n_numeric_out,
    }


# ── Public: Load Fitted Preprocessor ──────────────────────────────────────────

def load_preprocessor(session_id: str) -> Optional[ColumnTransformer]:
    """
    Load the fitted ColumnTransformer pipeline for an existing session.
    Returns None if not found (allows graceful fallback to old scaler.pkl).
    """
    path = get_session_file_path(session_id, "preprocessor.pkl")
    if path.exists():
        try:
            return joblib.load(path)
        except Exception:
            return None
    return None


def load_feature_names_out(session_id: str) -> Optional[list[str]]:
    """Load the stored feature_names_out list for a session."""
    from session_manager import load_json
    return load_json(session_id, "feature_names_out.json")


# ── Public: Transform a Single Inference Row ──────────────────────────────────

def transform_single_row(
    raw_features: dict[str, Any],
    session_id: str,
) -> tuple[np.ndarray, list[str]]:
    """
    Transform a single incoming feature dict through the fitted preprocessor.

    - Loads preprocessor.pkl and feature_names_out.json from session dir
    - Handles mixed types (str, int, float, None) gracefully
    - Returns (X_transformed, feature_names_out) ready for model.predict_proba()

    Raises:
        FileNotFoundError  - if preprocessor.pkl is missing (session not trained yet)
    """
    ct = load_preprocessor(session_id)

    if ct is None:
        # Backwards-compat: if old session has only scaler.pkl, fall through
        raise FileNotFoundError(
            f"preprocessor.pkl not found for session {session_id}. "
            "Session may have been trained before ColumnTransformer was added. "
            "Please retrain the model."
        )

    feature_names_out = load_feature_names_out(session_id) or []

    # Build a 1-row DataFrame matching the original feature schema
    col_type_map: dict[str, str] = {}
    from session_manager import load_json
    col_type_map = load_json(session_id, "col_type_map.json") or {}

    # All original feature cols (in order expected by the fitted CT)
    original_cols = list(col_type_map.keys())

    row_data: dict[str, Any] = {}
    for col in original_cols:
        val = raw_features.get(col, None)
        row_data[col] = val

    row_df = pd.DataFrame([row_data], columns=original_cols)

    # Ensure string columns stay as strings (not converted to NaN by pandas)
    for col, ctype in col_type_map.items():
        if ctype in ("low_card_cat", "high_card_cat"):
            row_df[col] = row_df[col].astype(str).replace("None", np.nan).replace("nan", np.nan)

    # Transform — TargetEncoder.transform() works without y
    try:
        X_transformed = ct.transform(row_df)
    except Exception as e:
        raise ValueError(
            f"Failed to transform inference row: {e}. "
            f"Make sure all feature types match the training schema."
        ) from e

    return X_transformed, feature_names_out
