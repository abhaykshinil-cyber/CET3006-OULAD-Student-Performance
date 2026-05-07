# =============================================================================
# data_loader.py — Feature engineering, preprocessing, fold management
# =============================================================================
#
# [WARN]  Data-leakage policy (strictly enforced):
#   • VLE / assessment aggregations are per-student historical summaries
#     computed from raw interaction logs — they do NOT use the target.
#   • LabelEncoding and StandardScaling are ALWAYS fit on the training
#     portion of each fold, then applied to the test portion.
#   • No transformation is ever fit on the full dataset before splitting.

import json
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    STUDENT_INFO_PATH, STUDENT_VLE_PATH,
    STUDENT_ASSESSMENT_PATH, ASSESSMENTS_PATH,
    FOLD_INDICES_PATH, RANDOM_SEED, N_FOLDS,
    safe_print,
)


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

def engineer_features(
    info_path=STUDENT_INFO_PATH,
    vle_path=STUDENT_VLE_PATH,
    assessment_path=STUDENT_ASSESSMENT_PATH,
    assessments_path=ASSESSMENTS_PATH,
    verbose=True,
):
    """
    Load OULAD CSVs and return an un-encoded feature DataFrame + binary target.

    Features added on top of the original studentInfo columns
    ----------------------------------------------------------
    From studentVle (aggregated per student × module × presentation):
      total_clicks      – total VLE interactions
      early_clicks      – clicks in the first 30 days  (early engagement proxy)
      late_clicks       – clicks after day 30          (sustained engagement)
      num_active_days   – distinct days with at least one click
      unique_activities – distinct VLE activity sites visited
      avg_daily_clicks  – total_clicks / num_active_days  (intensity proxy)

    From studentAssessment + assessments (aggregated per student):
      avg_score         – mean score across submitted assessments
      max_score         – highest individual assessment score
      num_submissions   – number of assessments submitted

    Returns
    -------
    X          : pd.DataFrame  (reset integer index; categorical cols are str)
    y          : pd.Series     (binary: Pass/Distinction=1, Fail/Withdrawn=0)
    cat_cols   : list[str]     (originally-categorical columns, still as str)
    cont_cols  : list[str]     (numeric columns)
    """
    if verbose:
        safe_print("Loading raw data files...")

    info_df    = pd.read_csv(info_path)
    vle_df     = pd.read_csv(vle_path)
    sa_df      = pd.read_csv(assessment_path)   # studentAssessment
    asmnt_df   = pd.read_csv(assessments_path)  # assessment metadata

    merge_keys = ["id_student", "code_module", "code_presentation"]

    # -- VLE aggregations ------------------------------------------------------

    vle_total = (
        vle_df.groupby(merge_keys)["sum_click"]
        .sum().reset_index()
        .rename(columns={"sum_click": "total_clicks"})
    )

    vle_early = (
        vle_df[vle_df["date"] <= 30]
        .groupby(merge_keys)["sum_click"]
        .sum().reset_index()
        .rename(columns={"sum_click": "early_clicks"})
    )

    vle_late = (
        vle_df[vle_df["date"] > 30]
        .groupby(merge_keys)["sum_click"]
        .sum().reset_index()
        .rename(columns={"sum_click": "late_clicks"})
    )

    vle_active = (
        vle_df[vle_df["sum_click"] > 0]
        .groupby(merge_keys)["date"]
        .nunique().reset_index()
        .rename(columns={"date": "num_active_days"})
    )

    vle_unique = (
        vle_df.groupby(merge_keys)["id_site"]
        .nunique().reset_index()
        .rename(columns={"id_site": "unique_activities"})
    )

    # -- Assessment aggregations -----------------------------------------------

    # Merge with metadata to get code_module / code_presentation
    sa_merged = pd.merge(sa_df, asmnt_df, on="id_assessment", how="left")
    sa_merged = sa_merged.dropna(subset=["score"])

    asmnt_feats = (
        sa_merged.groupby(merge_keys)["score"]
        .agg(avg_score="mean", max_score="max", num_submissions="count")
        .reset_index()
    )

    # -- Merge everything into studentInfo -------------------------------------

    df = info_df.copy()
    for feat_df in [vle_total, vle_early, vle_late, vle_active,
                    vle_unique, asmnt_feats]:
        df = pd.merge(df, feat_df, on=merge_keys, how="left")

    # Students with no VLE/assessment records -> fill with 0
    fill_map = {
        "total_clicks": 0, "early_clicks": 0, "late_clicks": 0,
        "num_active_days": 0, "unique_activities": 0,
        "avg_score": 0.0, "max_score": 0.0, "num_submissions": 0,
    }
    df = df.fillna(fill_map)

    # Derived: average click intensity (avoid div-by-zero)
    df["avg_daily_clicks"] = np.where(
        df["num_active_days"] > 0,
        df["total_clicks"] / df["num_active_days"],
        0.0,
    )

    # -- Target ----------------------------------------------------------------

    df["final_result"] = df["final_result"].map(
        {"Pass": 1, "Distinction": 1, "Fail": 0, "Withdrawn": 0}
    )
    df = df.dropna(subset=["final_result"])

    # Drop student/module identifiers (not predictive features)
    df = df.drop(columns=merge_keys, errors="ignore")

    # -- Separate target and features ------------------------------------------

    y = df["final_result"].astype(int).reset_index(drop=True)
    X = df.drop(columns=["final_result"]).reset_index(drop=True)

    cat_cols  = X.select_dtypes(include=["object"]).columns.tolist()
    cont_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    if verbose:
        safe_print(f"  Rows             : {X.shape[0]}")
        safe_print(f"  Features         : {X.shape[1]}  "
              f"({len(cat_cols)} cat, {len(cont_cols)} cont)")
        safe_print(f"  Categorical      : {cat_cols}")
        safe_print(f"  Continuous       : {cont_cols}")
        safe_print(f"  Positive rate    : {y.mean():.3f}")

    return X, y, cat_cols, cont_cols


# =============================================================================
# WITHIN-FOLD PREPROCESSING  (no leakage)
# =============================================================================

def preprocess_fold(X_train_raw, X_test_raw, cat_cols, cont_cols,
                    scale=False):
    """
    Fit encoders / scalers on the TRAINING portion of one fold only,
    then apply to the test portion.

    Parameters
    ----------
    X_train_raw, X_test_raw : pd.DataFrame
        Raw (un-encoded) feature frames for this fold.
    cat_cols : list[str]
    cont_cols : list[str]
    scale : bool
        Apply StandardScaler to continuous columns when True
        (used by TabNet and FT-Transformer).

    Returns
    -------
    X_tr, X_te : np.ndarray  (float32)
    encoders   : dict  {col_name: fitted LabelEncoder}
    scaler     : fitted StandardScaler or None
    """
    X_tr = X_train_raw.copy()
    X_te = X_test_raw.copy()

    # -- Label-encode categoricals ---------------------------------------------
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))

        # Map test values; unseen categories fall back to 0
        known = {cls: idx for idx, cls in enumerate(le.classes_)}
        X_te[col] = (
            X_te[col].astype(str)
            .map(known)
            .fillna(0)
            .astype(int)
        )
        encoders[col] = le

    # -- Optionally scale continuous features ----------------------------------
    scaler = None
    if scale and cont_cols:
        scaler = StandardScaler()
        X_tr[cont_cols] = scaler.fit_transform(X_tr[cont_cols])
        X_te[cont_cols] = scaler.transform(X_te[cont_cols])

    return (
        X_tr.values.astype(np.float32),
        X_te.values.astype(np.float32),
        encoders,
        scaler,
    )


def preprocess_fold_df(X_train_raw, X_test_raw, cat_cols, cont_cols,
                       scale=False):
    """
    Same as preprocess_fold but returns DataFrames (needed by FT-Transformer).
    """
    X_tr = X_train_raw.copy()
    X_te = X_test_raw.copy()

    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        X_tr[col] = le.fit_transform(X_tr[col].astype(str))
        known = {cls: idx for idx, cls in enumerate(le.classes_)}
        X_te[col] = (
            X_te[col].astype(str)
            .map(known)
            .fillna(0)
            .astype(int)
        )
        encoders[col] = le

    scaler = None
    if scale and cont_cols:
        scaler = StandardScaler()
        X_tr[cont_cols] = scaler.fit_transform(X_tr[cont_cols])
        X_te[cont_cols] = scaler.transform(X_te[cont_cols])

    # Enforce dtypes expected by pytorch-tabular
    for col in cat_cols:
        X_tr[col] = X_tr[col].astype(int)
        X_te[col] = X_te[col].astype(int)
    for col in cont_cols:
        X_tr[col] = X_tr[col].astype(float)
        X_te[col] = X_te[col].astype(float)

    return X_tr, X_te, encoders, scaler


# =============================================================================
# FOLD MANAGEMENT
# =============================================================================

def create_and_save_folds(X, y, n_splits=N_FOLDS, seed=RANDOM_SEED):
    """
    Create Stratified K-Fold splits and persist them to JSON.
    Returns list of (train_indices, test_indices) tuples.
    All subsequent models load from this file to guarantee identical splits.
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    folds = []
    for train_idx, test_idx in skf.split(X, y):
        folds.append({
            "train": train_idx.tolist(),
            "test":  test_idx.tolist(),
        })

    with open(FOLD_INDICES_PATH, "w") as f:
        json.dump(folds, f, indent=2)

    safe_print(f"Saved {n_splits}-fold indices -> {FOLD_INDICES_PATH}")
    return [(fold["train"], fold["test"]) for fold in folds]


def load_folds():
    """Load persisted fold indices (raises FileNotFoundError if absent)."""
    with open(FOLD_INDICES_PATH, "r") as f:
        raw = json.load(f)
    return [(fold["train"], fold["test"]) for fold in raw]


def get_or_create_folds(X, y):
    """Load existing folds or create them if the file does not exist yet."""
    try:
        folds = load_folds()
        safe_print(f"Loaded existing fold splits from {FOLD_INDICES_PATH}.")
    except FileNotFoundError:
        safe_print("No fold file found — generating new stratified splits...")
        folds = create_and_save_folds(X, y)
    return folds


# =============================================================================
# RESULT I/O HELPERS
# =============================================================================

def save_results(results_dict, path):
    """Persist a results dictionary to JSON (pretty-printed)."""
    with open(path, "w") as f:
        json.dump(results_dict, f, indent=2)
    safe_print(f"  [OK] Saved -> {path}")


def load_results(path):
    """Load a results dictionary from JSON."""
    with open(path, "r") as f:
        return json.load(f)
