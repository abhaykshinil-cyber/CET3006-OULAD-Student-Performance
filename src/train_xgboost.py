# =============================================================================
# train_xgboost.py  --  XGBoost: independent CV + Optuna + importance + SHAP
# =============================================================================
#
# Run standalone:
#   python train_xgboost.py           # 5-fold CV with default params
#   python train_xgboost.py --tune    # CV + Optuna tuning
#
# Requires:  pip install xgboost shap optuna

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import xgboost as xgb
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    RESULTS_DIR, XGB_RESULTS_PATH, SHAP_VALUES_PATH,
    OPTUNA_TRIALS, RANDOM_SEED, safe_print,
)
from data_loader import (
    engineer_features,
    preprocess_fold,
    get_or_create_folds,
    save_results,
)

MODEL_NAME = "XGBoost"

# NOTE: use_label_encoder was removed in XGBoost 2.0 -- never include it.
# early_stopping_rounds lives in the constructor so it works for XGBoost 1.x and 2.x.
DEFAULT_PARAMS = {
    "n_estimators":          300,
    "learning_rate":         0.05,
    "max_depth":             6,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "eval_metric":           "logloss",
    "early_stopping_rounds": 20,
    "verbosity":             0,
    "random_state":          RANDOM_SEED,
}

EARLY_STOP_ROUNDS = 20   # module-level constant used by Optuna / importance helpers


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cv(X, y, cat_cols, cont_cols, folds, params=None):
    """
    5-fold stratified CV for XGBoost.

    Early stopping is driven by an inner validation split (10 % of the
    training fold) so the outer test fold remains completely unseen.
    early_stopping_rounds is expected inside `params` / DEFAULT_PARAMS;
    it must NOT also be passed to .fit() to avoid the XGBoost 2.x warning.
    """
    base  = dict(params or DEFAULT_PARAMS)
    y_arr = y.values
    fold_accs, fold_f1s, fold_aucs = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        safe_print(f"  Fold {fold_idx + 1}/{len(folds)} ...", end=" ", flush=True)

        X_tr_raw  = X.iloc[train_idx]
        X_te_raw  = X.iloc[test_idx]
        y_tr_full = y_arr[train_idx]
        y_te      = y_arr[test_idx]

        # Encode: fit on full training fold, apply to test fold
        X_tr_enc, X_te_enc, _, _ = preprocess_fold(
            X_tr_raw, X_te_raw, cat_cols, cont_cols, scale=False,
        )

        # Carve inner val from training encoding (no extra preprocess call needed)
        n_val      = max(1, int(len(train_idx) * 0.10))
        X_va_enc   = X_tr_enc[:n_val]
        X_itr_enc  = X_tr_enc[n_val:]
        y_inner_va = y_tr_full[:n_val]
        y_inner_tr = y_tr_full[n_val:]

        # early_stopping_rounds is inside `base` -- not passed to .fit()
        model = xgb.XGBClassifier(**base)
        model.fit(
            X_itr_enc, y_inner_tr,
            eval_set=[(X_va_enc, y_inner_va)],
            verbose=False,
        )

        preds = model.predict(X_te_enc)
        proba = model.predict_proba(X_te_enc)[:, 1]

        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, zero_division=0)
        auc = roc_auc_score(y_te, proba)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_aucs.append(auc)
        safe_print(f"Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

    return {
        "fold_accuracies": fold_accs,
        "fold_f1s":        fold_f1s,
        "fold_aucs":       fold_aucs,
        "mean_accuracy":   float(np.mean(fold_accs)),
        "std_accuracy":    float(np.std(fold_accs)),
        "mean_f1":         float(np.mean(fold_f1s)),
        "std_f1":          float(np.std(fold_f1s)),
        "mean_auc":        float(np.mean(fold_aucs)),
        "std_auc":         float(np.std(fold_aucs)),
    }


# =============================================================================
# HYPERPARAMETER TUNING  (Optuna -- fold-0 split, lightweight)
# =============================================================================

def run_optuna(X, y, cat_cols, cont_cols, folds):
    """
    Tune XGBoost with Optuna on fold-0.
    Encodes once and reuses array slices so preprocessing is done only once.
    """
    safe_print(f"\n  Running Optuna ({OPTUNA_TRIALS} trials, fold-0 split) ...")

    train_idx, val_idx = folds[0]
    X_tr_raw = X.iloc[train_idx]
    X_va_raw = X.iloc[val_idx]
    y_arr    = y.values
    y_tr     = y_arr[train_idx]
    y_va     = y_arr[val_idx]

    # Encode once: fit on full training fold, produce both train and val encodings
    X_tr_enc, X_va_enc, _, _ = preprocess_fold(
        X_tr_raw, X_va_raw, cat_cols, cont_cols, scale=False,
    )

    # Inner val carved from training encoding (for early stopping inside each trial)
    n_inner  = max(1, int(len(train_idx) * 0.10))
    X_iv_enc = X_tr_enc[:n_inner]   # inner val
    X_i_enc  = X_tr_enc[n_inner:]   # inner training
    y_iv     = y_tr[:n_inner]
    y_i      = y_tr[n_inner:]

    def objective(trial):
        p = {
            "n_estimators":          trial.suggest_int("n_estimators", 100, 500),
            "learning_rate":         trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "max_depth":             trial.suggest_int("max_depth", 3, 10),
            "subsample":             trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree":      trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight":      trial.suggest_int("min_child_weight", 1, 10),
            "gamma":                 trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha":             trial.suggest_float("reg_alpha", 0.0, 5.0),
            "reg_lambda":            trial.suggest_float("reg_lambda", 0.0, 5.0),
            "eval_metric":           "logloss",
            "early_stopping_rounds": EARLY_STOP_ROUNDS,
            "verbosity":             0,
            "random_state":          RANDOM_SEED,
        }
        model = xgb.XGBClassifier(**p)
        model.fit(
            X_i_enc, y_i,
            eval_set=[(X_iv_enc, y_iv)],
            verbose=False,
        )
        preds = model.predict(X_va_enc)
        return f1_score(y_va, preds, zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_params
    best.update({
        "eval_metric":           "logloss",
        "early_stopping_rounds": EARLY_STOP_ROUNDS,
        "verbosity":             0,
        "random_state":          RANDOM_SEED,
    })
    safe_print(f"  Best val F1  : {study.best_value:.4f}")
    safe_print(f"  Best params  : {best}")
    return best, float(study.best_value)


# =============================================================================
# FEATURE IMPORTANCE  (XGBoost gain)
# =============================================================================

def compute_feature_importance(X, y, cat_cols, cont_cols, folds, params=None):
    """
    Fit XGBoost on fold-0 training split and return ranked gain importance.
    Returns (importance_df, trained_model, X_va_enc) where X_va_enc is the
    fold-0 test encoding used downstream for SHAP analysis.
    """
    base = dict(params or DEFAULT_PARAMS)
    # Guard: callers may pass tuned params that omit early_stopping_rounds
    base.setdefault("early_stopping_rounds", EARLY_STOP_ROUNDS)

    train_idx, val_idx = folds[0]
    X_tr_enc, X_va_enc, _, _ = preprocess_fold(
        X.iloc[train_idx], X.iloc[val_idx],
        cat_cols, cont_cols, scale=False,
    )
    y_arr = y.values
    y_tr  = y_arr[train_idx]

    # Inner val for early stopping
    n_inner  = max(1, int(len(train_idx) * 0.10))
    X_iv_enc = X_tr_enc[:n_inner]
    X_it_enc = X_tr_enc[n_inner:]
    y_iv     = y_tr[:n_inner]
    y_it     = y_tr[n_inner:]

    model = xgb.XGBClassifier(**base)
    model.fit(
        X_it_enc, y_it,
        eval_set=[(X_iv_enc, y_iv)],
        verbose=False,
    )

    importance_df = (
        pd.DataFrame({
            "feature":    X.columns.tolist(),
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df, model, X_va_enc


# =============================================================================
# SHAP ANALYSIS
# =============================================================================

def compute_shap(model, X_encoded, feature_names):
    """
    Compute SHAP values with TreeExplainer.

    Returns
    -------
    shap_vals   : np.ndarray  (n_samples, n_features)  -- class-1 SHAP values
    mean_abs_df : pd.DataFrame  ranked by mean |SHAP| per feature
    """
    explainer = shap.TreeExplainer(model)
    shap_raw  = explainer.shap_values(X_encoded)

    # Binary XGBoost returns a 2-D array directly; older versions return a list
    shap_vals = shap_raw[1] if isinstance(shap_raw, list) else shap_raw

    mean_abs_df = (
        pd.DataFrame({
            "feature":       feature_names,
            "mean_abs_shap": np.abs(shap_vals).mean(axis=0),
        })
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return shap_vals, mean_abs_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="XGBoost training pipeline")
    parser.add_argument("--tune",  action="store_true",
                        help="Run Optuna hyperparameter tuning")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation step")
    args = parser.parse_args()

    safe_print("=" * 65)
    safe_print(f"  {MODEL_NAME} -- Independent Training Pipeline")
    safe_print("=" * 65)

    # ---- 1. Load data and folds ---------------------------------------------
    X, y, cat_cols, cont_cols = engineer_features()
    folds   = get_or_create_folds(X, y)
    results = {"model": MODEL_NAME}

    # ---- 2. Cross-validation ------------------------------------------------
    if not args.no_cv:
        safe_print("\n[Step 1]  5-Fold Stratified CV  (default params)")
        cv = run_cv(X, y, cat_cols, cont_cols, folds)
        results["cv"] = cv
        safe_print(f"\n  Mean Accuracy : {cv['mean_accuracy']:.4f} +/- {cv['std_accuracy']:.4f}")
        safe_print(f"  Mean F1-score : {cv['mean_f1']:.4f} +/- {cv['std_f1']:.4f}")
        safe_print(f"  Mean ROC-AUC  : {cv['mean_auc']:.4f} +/- {cv['std_auc']:.4f}")

    # ---- 3. Optuna tuning (optional) ----------------------------------------
    best_params = None
    if args.tune:
        safe_print("\n[Step 2]  Optuna Hyperparameter Tuning")
        best_params, best_val_f1 = run_optuna(X, y, cat_cols, cont_cols, folds)
        results["best_params"] = best_params
        results["best_val_f1"] = best_val_f1

        safe_print("\n[Step 2b] CV with tuned params")
        tuned_cv = run_cv(X, y, cat_cols, cont_cols, folds, params=best_params)
        results["tuned_cv"] = tuned_cv
        safe_print(f"\n  Tuned Mean F1 : {tuned_cv['mean_f1']:.4f} +/- {tuned_cv['std_f1']:.4f}")

    # ---- 4. Feature importance + SHAP ---------------------------------------
    safe_print("\n[Step 3]  Feature Importance (XGBoost gain)")
    fi_params = best_params if best_params else DEFAULT_PARAMS
    importance_df, trained_model, X_va_enc = compute_feature_importance(
        X, y, cat_cols, cont_cols, folds, fi_params,
    )
    results["feature_importance"] = importance_df.to_dict(orient="records")

    fi_path = os.path.join(RESULTS_DIR, "feature_importance_xgb.csv")
    importance_df.to_csv(fi_path, index=False)
    safe_print("\n  Top 10 Features (XGBoost Gain):")
    safe_print(importance_df.head(10).to_string(index=False))
    safe_print(f"\n  Feature importance saved -> {fi_path}")

    safe_print("\n[Step 4]  SHAP Analysis")
    shap_vals, shap_df = compute_shap(trained_model, X_va_enc, X.columns.tolist())

    np.save(SHAP_VALUES_PATH, shap_vals)
    shap_csv = os.path.join(RESULTS_DIR, "shap_summary.csv")
    shap_df.to_csv(shap_csv, index=False)
    results["shap_summary"] = shap_df.to_dict(orient="records")

    safe_print("\n  Top 10 Features (Mean |SHAP|):")
    safe_print(shap_df.head(10).to_string(index=False))
    safe_print(f"\n  SHAP array saved  -> {SHAP_VALUES_PATH}")
    safe_print(f"  SHAP summary CSV  -> {shap_csv}")

    # ---- 5. Save all results ------------------------------------------------
    save_results(results, XGB_RESULTS_PATH)

    safe_print(f"\n{'=' * 65}")
    safe_print(f"  {MODEL_NAME} complete.")
    safe_print(f"  Results -> {XGB_RESULTS_PATH}")
    safe_print(f"{'=' * 65}")
