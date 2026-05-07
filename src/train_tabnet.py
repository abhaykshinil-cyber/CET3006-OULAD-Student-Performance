# =============================================================================
# train_tabnet.py — TabNet: independent CV + Optuna with early stopping
# =============================================================================
#
# Run standalone:
#   python train_tabnet.py            # 5-fold CV with default params (CPU)
#   python train_tabnet.py --tune     # CV + Optuna tuning
#   python train_tabnet.py --gpu      # use GPU
#
# Requires:  pip install pytorch-tabnet optuna

import os
import sys
import gc
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from pytorch_tabnet.tab_model import TabNetClassifier
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    TN_RESULTS_PATH, OPTUNA_TRIALS, RANDOM_SEED,
    TABNET_MAX_EPOCHS, TABNET_PATIENCE, VAL_FRACTION, safe_print,
)
from data_loader import (
    engineer_features,
    preprocess_fold,
    get_or_create_folds,
    save_results,
)

MODEL_NAME = "TabNet"

DEFAULT_PARAMS = {
    "n_d":           32,
    "n_a":           32,
    "n_steps":        5,
    "gamma":          1.3,
    "n_independent":  2,
    "n_shared":       2,
    "lambda_sparse":  1e-3,
}

FIT_KWARGS = {
    "max_epochs":         TABNET_MAX_EPOCHS,
    "patience":           TABNET_PATIENCE,
    "batch_size":         1024,
    "virtual_batch_size": 128,
}


# =============================================================================
# MODEL FACTORY
# =============================================================================

def build_tabnet(params=None, device="cpu"):
    """Instantiate a TabNetClassifier from an optional params dict."""
    p = params or DEFAULT_PARAMS
    return TabNetClassifier(
        n_d=p.get("n_d", 32),
        n_a=p.get("n_a", 32),
        n_steps=p.get("n_steps", 5),
        gamma=p.get("gamma", 1.3),
        n_independent=p.get("n_independent", 2),
        n_shared=p.get("n_shared", 2),
        lambda_sparse=p.get("lambda_sparse", 1e-3),
        device_name=device,
        seed=RANDOM_SEED,
        verbose=0,
    )


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cv(X, y, cat_cols, cont_cols, folds, params=None, device="cpu"):
    """
    5-fold stratified CV for TabNet.

    Strategy inside each fold
    -------------------------
    • Preprocess (scale) on training portion only — no leakage.
    • Carve a small inner-validation set (VAL_FRACTION) from the training
      portion to drive early stopping.
    • Evaluate on the held-out test portion.
    """
    y_arr = y.values
    fold_accs, fold_f1s, fold_aucs = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        safe_print(f"  Fold {fold_idx + 1}/{len(folds)} ...", end=" ", flush=True)

        # Encode / scale: fit on training fold, apply to test
        X_tr_enc, X_te_enc, _, _ = preprocess_fold(
            X.iloc[train_idx], X.iloc[test_idx],
            cat_cols, cont_cols, scale=True,
        )
        y_tr = y_arr[train_idx]
        y_te = y_arr[test_idx]

        # Inner validation split for early stopping (from training fold)
        X_itr, X_iva, y_itr, y_iva = train_test_split(
            X_tr_enc, y_tr,
            test_size=VAL_FRACTION,
            random_state=RANDOM_SEED,
            stratify=y_tr,
        )

        model = build_tabnet(params, device)
        model.fit(
            X_train=X_itr, y_train=y_itr,
            eval_set=[(X_iva, y_iva)],
            **FIT_KWARGS,
        )

        preds = model.predict(X_te_enc).reshape(-1)
        proba = model.predict_proba(X_te_enc)[:, 1]

        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, zero_division=0)
        auc = roc_auc_score(y_te, proba)

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_aucs.append(auc)
        safe_print(f"Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

        del model
        gc.collect()

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
# HYPERPARAMETER TUNING  (Optuna — fold-0, lightweight)
# =============================================================================

def run_optuna(X, y, cat_cols, cont_cols, folds, device="cpu"):
    """
    Tune TabNet architecture hyperparameters using Optuna on fold-0.
    Uses an inner-val split for early stopping within each trial.
    """
    safe_print(f"\n  Running Optuna ({OPTUNA_TRIALS} trials, fold-0 split) ...")

    train_idx, val_idx = folds[0]
    X_tr_enc, X_va_enc, _, _ = preprocess_fold(
        X.iloc[train_idx], X.iloc[val_idx],
        cat_cols, cont_cols, scale=True,
    )
    y_arr    = y.values
    y_tr     = y_arr[train_idx]
    y_va     = y_arr[val_idx]

    # Inner val from training for early stopping
    X_itr, X_iva, y_itr, y_iva = train_test_split(
        X_tr_enc, y_tr,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=y_tr,
    )

    def objective(trial):
        p = {
            "n_d":           trial.suggest_int("n_d", 8, 64),
            "n_a":           trial.suggest_int("n_a", 8, 64),
            "n_steps":       trial.suggest_int("n_steps", 3, 10),
            "gamma":         trial.suggest_float("gamma", 1.0, 2.0),
            "n_independent": trial.suggest_int("n_independent", 1, 5),
            "n_shared":      trial.suggest_int("n_shared", 1, 5),
            "lambda_sparse": trial.suggest_float("lambda_sparse", 1e-6, 1e-2, log=True),
        }
        model = build_tabnet(p, device)
        model.fit(
            X_train=X_itr, y_train=y_itr,
            eval_set=[(X_iva, y_iva)],
            **FIT_KWARGS,
        )
        preds = model.predict(X_va_enc).reshape(-1)
        score = f1_score(y_va, preds, zero_division=0)
        del model
        gc.collect()
        return score

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=OPTUNA_TRIALS, show_progress_bar=False)

    best = study.best_params
    safe_print(f"  Best val F1   : {study.best_value:.4f}")
    safe_print(f"  Best params   : {best}")
    return best, float(study.best_value)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TabNet training pipeline")
    parser.add_argument("--tune",  action="store_true",
                        help="Run Optuna hyperparameter tuning")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation step")
    parser.add_argument("--gpu",   action="store_true",
                        help="Use GPU (CUDA) device")
    args   = parser.parse_args()
    device = "cuda" if args.gpu else "cpu"

    safe_print("=" * 65)
    safe_print(f"  {MODEL_NAME} — Independent Training Pipeline  [{device.upper()}]")
    safe_print("=" * 65)

    # -- 1. Load data & folds --------------------------------------------------
    X, y, cat_cols, cont_cols = engineer_features()
    folds   = get_or_create_folds(X, y)
    results = {"model": MODEL_NAME, "device": device}

    # -- 2. Cross-validation ---------------------------------------------------
    if not args.no_cv:
        safe_print(f"\n[Step 1]  5-Fold Stratified CV  (default params)")
        cv = run_cv(X, y, cat_cols, cont_cols, folds, device=device)
        results["cv"] = cv
        safe_print(f"\n  Mean Accuracy : {cv['mean_accuracy']:.4f} +/- {cv['std_accuracy']:.4f}")
        safe_print(f"  Mean F1-score : {cv['mean_f1']:.4f} +/- {cv['std_f1']:.4f}")
        safe_print(f"  Mean ROC-AUC  : {cv['mean_auc']:.4f} +/- {cv['std_auc']:.4f}")

    # -- 3. Optuna tuning (optional) -------------------------------------------
    best_params = None
    if args.tune:
        safe_print("\n[Step 2]  Optuna Hyperparameter Tuning")
        best_params, best_val_f1 = run_optuna(X, y, cat_cols, cont_cols, folds, device)
        results["best_params"] = best_params
        results["best_val_f1"] = best_val_f1

        safe_print("\n[Step 2b] CV with tuned params")
        tuned_cv = run_cv(X, y, cat_cols, cont_cols, folds,
                          params=best_params, device=device)
        results["tuned_cv"] = tuned_cv
        safe_print(f"\n  Tuned Mean F1 : {tuned_cv['mean_f1']:.4f} +/- {tuned_cv['std_f1']:.4f}")

    # -- 4. Save results -------------------------------------------------------
    save_results(results, TN_RESULTS_PATH)

    safe_print(f"\n{'=' * 65}")
    safe_print(f"  {MODEL_NAME} complete.")
    safe_print(f"  Results  -> {TN_RESULTS_PATH}")
    safe_print(f"{'=' * 65}")
