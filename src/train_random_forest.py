# =============================================================================
# train_random_forest.py — Random Forest: independent CV + tuning + importance
# =============================================================================
#
# Run standalone:
#   python train_random_forest.py           # 5-fold CV with default params
#   python train_random_forest.py --tune    # CV + Optuna tuning
#   python train_random_forest.py --no-cv  # skip CV (tuning only)

import os
import sys
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import RESULTS_DIR, RF_RESULTS_PATH, OPTUNA_TRIALS, RANDOM_SEED, safe_print
from data_loader import (
    engineer_features,
    preprocess_fold,
    get_or_create_folds,
    save_results,
)

MODEL_NAME = "Random Forest"

DEFAULT_PARAMS = {
    "n_estimators": 100,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cv(X, y, cat_cols, cont_cols, folds, params=None):
    """
    Run 5-fold stratified CV for Random Forest.

    Parameters
    ----------
    X, y         : DataFrame / Series  (full dataset, reset index)
    cat_cols     : list[str]
    cont_cols    : list[str]
    folds        : list of (train_idx, test_idx) — SHARED across all models
    params       : dict  RF hyperparameters (default: DEFAULT_PARAMS)

    Returns
    -------
    dict  with per-fold scores and aggregate mean/std
    """
    if params is None:
        params = DEFAULT_PARAMS

    y_arr = y.values
    fold_accs, fold_f1s, fold_aucs = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        safe_print(f"  Fold {fold_idx + 1}/{len(folds)} ...", end=" ", flush=True)

        # -- Leakage-safe preprocessing: fit on train, apply to test ----------
        X_tr, X_te, _, _ = preprocess_fold(
            X.iloc[train_idx], X.iloc[test_idx],
            cat_cols, cont_cols, scale=False,   # trees don't need scaling
        )
        y_tr, y_te = y_arr[train_idx], y_arr[test_idx]

        model = RandomForestClassifier(**params)
        model.fit(X_tr, y_tr)

        preds = model.predict(X_te)
        proba = model.predict_proba(X_te)[:, 1]

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
# HYPERPARAMETER TUNING  (Optuna — lightweight, fold-0 only)
# =============================================================================

def run_optuna(X, y, cat_cols, cont_cols, folds):
    """
    Tune RF using Optuna on fold-0 train / val split.
    Keeps tuning fast by using a single fold rather than nested CV.
    """
    safe_print(f"\n  Running Optuna ({OPTUNA_TRIALS} trials, fold-0 split) ...")

    train_idx, val_idx = folds[0]
    X_tr, X_va, _, _ = preprocess_fold(
        X.iloc[train_idx], X.iloc[val_idx],
        cat_cols, cont_cols, scale=False,
    )
    y_arr        = y.values
    y_tr, y_va   = y_arr[train_idx], y_arr[val_idx]

    def objective(trial):
        p = {
            "n_estimators":      trial.suggest_int("n_estimators", 50, 400),
            "max_depth":         trial.suggest_int("max_depth", 3, 25),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf":  trial.suggest_int("min_samples_leaf", 1, 10),
            "max_features":      trial.suggest_categorical(
                                    "max_features", ["sqrt", "log2"]),
            "random_state":      RANDOM_SEED,
            "n_jobs":            -1,
        }
        model = RandomForestClassifier(**p)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        return f1_score(y_va, preds, zero_division=0)

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_params
    best.update({"random_state": RANDOM_SEED, "n_jobs": -1})
    safe_print(f"  Best val F1   : {study.best_value:.4f}")
    safe_print(f"  Best params   : {best}")
    return best, float(study.best_value)


# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

def compute_feature_importance(X, y, cat_cols, cont_cols, folds, params=None):
    """
    Fit RF on fold-0 training set and return a ranked importance DataFrame.
    Uses a higher n_estimators than default for more stable importances.
    """
    if params is None:
        params = {**DEFAULT_PARAMS, "n_estimators": 200}

    train_idx, val_idx = folds[0]
    X_tr, _, _, _ = preprocess_fold(
        X.iloc[train_idx], X.iloc[val_idx],
        cat_cols, cont_cols, scale=False,
    )
    y_tr = y.values[train_idx]

    model = RandomForestClassifier(**params)
    model.fit(X_tr, y_tr)

    importance_df = (
        pd.DataFrame({
            "feature":    X.columns.tolist(),
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return importance_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Random Forest training pipeline")
    parser.add_argument("--tune",  action="store_true",
                        help="Run Optuna hyperparameter tuning after CV")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation step")
    args = parser.parse_args()

    safe_print("=" * 65)
    safe_print(f"  {MODEL_NAME} — Independent Training Pipeline")
    safe_print("=" * 65)

    # -- 1. Load data & folds --------------------------------------------------
    X, y, cat_cols, cont_cols = engineer_features()
    folds   = get_or_create_folds(X, y)
    results = {"model": MODEL_NAME}

    # -- 2. Cross-validation ---------------------------------------------------
    if not args.no_cv:
        safe_print(f"\n[Step 1]  5-Fold Stratified CV  (default params)")
        cv = run_cv(X, y, cat_cols, cont_cols, folds)
        results["cv"] = cv
        safe_print(f"\n  Mean Accuracy : {cv['mean_accuracy']:.4f} +/- {cv['std_accuracy']:.4f}")
        safe_print(f"  Mean F1-score : {cv['mean_f1']:.4f} +/- {cv['std_f1']:.4f}")
        safe_print(f"  Mean ROC-AUC  : {cv['mean_auc']:.4f} +/- {cv['std_auc']:.4f}")

    # -- 3. Optuna tuning (optional) -------------------------------------------
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

    # -- 4. Feature importance -------------------------------------------------
    safe_print("\n[Step 3]  Feature Importance")
    fi_params    = best_params if best_params else {**DEFAULT_PARAMS, "n_estimators": 200}
    importance_df = compute_feature_importance(X, y, cat_cols, cont_cols, folds, fi_params)

    results["feature_importance"] = importance_df.to_dict(orient="records")

    fi_path = os.path.join(RESULTS_DIR, "feature_importance_rf.csv")
    importance_df.to_csv(fi_path, index=False)

    safe_print("\n  Top 10 Features (Mean Decrease Impurity):")
    safe_print(importance_df.head(10).to_string(index=False))
    safe_print(f"\n  Feature importance saved -> {fi_path}")

    # -- 5. Save all results ---------------------------------------------------
    save_results(results, RF_RESULTS_PATH)

    safe_print(f"\n{'=' * 65}")
    safe_print(f"  {MODEL_NAME} complete.")
    safe_print(f"  Results  -> {RF_RESULTS_PATH}")
    safe_print(f"{'=' * 65}")
