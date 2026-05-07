# =============================================================================
# train_ft_transformer.py — FT-Transformer: independent CV + Optuna
# =============================================================================
#
# Run standalone:
#   python train_ft_transformer.py           # 5-fold CV (CPU)
#   python train_ft_transformer.py --tune    # CV + Optuna tuning
#   python train_ft_transformer.py --gpu     # use GPU
#
# Requires:  pip install pytorch-tabular optuna

import os
import sys
import gc
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

# Handle pytorch-tabular version differences
try:
    from pytorch_tabular.models.ft_transformer.config import FTTransformerConfig
except ImportError:                         # pytorch-tabular >= 1.0
    from pytorch_tabular.models import FTTransformerConfig

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    FT_RESULTS_PATH, OPTUNA_TRIALS, RANDOM_SEED,
    FT_MAX_EPOCHS, FT_PATIENCE, VAL_FRACTION, safe_print,
)
from data_loader import (
    engineer_features,
    preprocess_fold_df,
    get_or_create_folds,
    save_results,
)

MODEL_NAME = "FT-Transformer"

DEFAULT_MODEL_KWARGS = {
    "input_embed_dim":  32,
    "num_heads":         4,
    "num_attn_blocks":   4,
    "attn_dropout":     0.1,
    "ff_dropout":       0.1,
    "learning_rate":    1e-3,
}


# =============================================================================
# CONFIG BUILDERS
# =============================================================================

def _data_config(cat_cols, cont_cols):
    return DataConfig(
        target=["target"],
        continuous_cols=cont_cols,
        categorical_cols=cat_cols,
        num_workers=0,          # safer on Windows
    )


def _trainer_config(accelerator="cpu", max_epochs=FT_MAX_EPOCHS, patience=FT_PATIENCE):
    """
    Build a TrainerConfig that works across pytorch-tabular versions.
    Checkpoints are disabled so CV does not leave artefacts on disk.
    """
    try:
        kw = {"accelerator": accelerator, "enable_model_summary": False}
        if accelerator == "gpu":
            kw["devices"] = 1
        return TrainerConfig(
            max_epochs=max_epochs,
            batch_size=512,
            early_stopping="valid_loss",
            early_stopping_patience=patience,
            checkpoints=None,
            load_best=True,
            progress_bar="none",
            trainer_kwargs=kw,
        )
    except TypeError:
        # Legacy pytorch-tabular < 1.0 API
        gpus = 0 if accelerator == "cpu" else 1
        return TrainerConfig(
            max_epochs=max_epochs,
            batch_size=512,
            gpus=gpus,
            early_stopping=True,
        )


def _build_ft_model(cat_cols, cont_cols, model_kwargs=None, accelerator="cpu"):
    """Return a fresh, untrained FT-Transformer TabularModel."""
    kw = model_kwargs or DEFAULT_MODEL_KWARGS

    # Ensure num_heads divides input_embed_dim (architectural constraint)
    embed  = kw.get("input_embed_dim", 32)
    heads  = kw.get("num_heads", 4)
    while embed % heads != 0 and heads > 1:
        heads //= 2
    kw = dict(kw)
    kw["num_heads"] = heads

    model_cfg = FTTransformerConfig(
        task="classification",
        input_embed_dim=embed,
        num_heads=heads,
        num_attn_blocks=kw.get("num_attn_blocks", 4),
        attn_dropout=kw.get("attn_dropout", 0.1),
        ff_dropout=kw.get("ff_dropout", 0.1),
        learning_rate=kw.get("learning_rate", 1e-3),
    )
    return TabularModel(
        data_config=_data_config(cat_cols, cont_cols),
        model_config=model_cfg,
        trainer_config=_trainer_config(accelerator),
        optimizer_config=OptimizerConfig(),
    )


# =============================================================================
# FOLD DATAFRAME BUILDER  (no leakage)
# =============================================================================

def _make_fold_dfs(X_tr_raw, X_te_raw, y_tr_arr, y_te_arr,
                   cat_cols, cont_cols):
    """
    Build pytorch-tabular ready DataFrames for one fold.

    Encoding is fit on X_tr_raw only (preprocess_fold_df).
    An inner validation split (VAL_FRACTION) is carved from the training
    DataFrame to drive early stopping.

    Returns
    -------
    tr_df  : DataFrame  training set (90 % of training fold) + "target"
    val_df : DataFrame  inner val (10 % of training fold) + "target"
    te_df  : DataFrame  test fold + "target"
    """
    X_tr_enc, X_te_enc, _, _ = preprocess_fold_df(
        X_tr_raw, X_te_raw, cat_cols, cont_cols, scale=True,
    )

    tr_df = X_tr_enc.copy()
    tr_df["target"] = y_tr_arr.astype("int64")

    te_df = X_te_enc.copy()
    te_df["target"] = y_te_arr.astype("int64")

    tr_split, val_split = train_test_split(
        tr_df,
        test_size=VAL_FRACTION,
        random_state=RANDOM_SEED,
        stratify=tr_df["target"],
    )
    return (
        tr_split.reset_index(drop=True),
        val_split.reset_index(drop=True),
        te_df,
    )


def _ft_predict(model, X_df_no_target):
    """Extract integer class predictions from pytorch-tabular output."""
    out = model.predict(X_df_no_target)
    return out["target_prediction"].astype(int).values


def _ft_proba(model, X_df_no_target):
    """
    Return a (n_samples, 2) probability matrix from a pytorch-tabular model.

    Column 0 = P(class 0), Column 1 = P(class 1).

    pytorch-tabular names probability columns as '0_target_probability' and
    '1_target_probability' (order varies by version).  We collect ALL prob
    columns, sort them so class-0 is at index 0 and class-1 at index 1, and
    return a numpy array so callers can consistently use proba[:, 1].

    Returns None when no probability columns are found so the caller can
    fall back to NaN for AUC rather than crashing.
    """
    try:
        out = model.predict(X_df_no_target)

        # Collect every column that carries a probability score.
        # Exclude 'target_prediction' (the hard class label) explicitly.
        prob_cols = [
            c for c in out.columns
            if "prob" in c.lower() and "prediction" not in c.lower()
        ]

        if not prob_cols:
            return None

        # Sort so class-0 column comes first, class-1 second.
        # e.g. ['0_target_probability', '1_target_probability'] -> already sorted.
        prob_cols_sorted = sorted(prob_cols)

        if len(prob_cols_sorted) >= 2:
            # Full (n_samples, 2) matrix — preferred path.
            probs = out[prob_cols_sorted].values.astype(float)
        else:
            # Only one column found; assume it is P(class 1) and derive P(class 0).
            p1 = out[prob_cols_sorted[0]].values.astype(float)
            probs = np.column_stack([1.0 - p1, p1])

        return probs          # shape (n_samples, 2)

    except Exception:
        pass
    return None               # caller will handle None -> skip AUC


# =============================================================================
# CROSS-VALIDATION
# =============================================================================

def run_cv(X, y, cat_cols, cont_cols, folds, model_kwargs=None, accelerator="cpu"):
    """
    5-fold stratified CV for FT-Transformer.

    A fresh model is built and trained for every fold to avoid any
    weight contamination between folds.
    """
    y_arr = y.values
    fold_accs, fold_f1s, fold_aucs = [], [], []

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        safe_print(f"  Fold {fold_idx + 1}/{len(folds)} ...", end=" ", flush=True)

        tr_df, val_df, te_df = _make_fold_dfs(
            X.iloc[train_idx], X.iloc[test_idx],
            y_arr[train_idx],  y_arr[test_idx],
            cat_cols, cont_cols,
        )
        X_te_notar = te_df.drop(columns=["target"])
        y_te       = y_arr[test_idx]

        model = _build_ft_model(cat_cols, cont_cols, model_kwargs, accelerator)
        model.fit(train=tr_df, validation=val_df)

        preds = _ft_predict(model, X_te_notar)
        proba = _ft_proba(model, X_te_notar)   # (n_samples, 2) or None

        acc = accuracy_score(y_te, preds)
        f1  = f1_score(y_te, preds, zero_division=0)
        if proba is not None:
            # proba[:, 1] is P(class=1) — consistent with RF and XGBoost
            print("DEBUG probs range:", proba[:, 1].min(), proba[:, 1].max())
            try:
                auc = float(roc_auc_score(y_te, proba[:, 1]))
            except Exception:
                auc = float("nan")
        else:
            auc = float("nan")

        fold_accs.append(acc)
        fold_f1s.append(f1)
        fold_aucs.append(auc)
        safe_print(f"Acc={acc:.4f}  F1={f1:.4f}  AUC={auc:.4f}")

        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return {
        "fold_accuracies": fold_accs,
        "fold_f1s":        fold_f1s,
        "fold_aucs":       fold_aucs,
        "mean_accuracy":   float(np.mean(fold_accs)),
        "std_accuracy":    float(np.std(fold_accs)),
        "mean_f1":         float(np.mean(fold_f1s)),
        "std_f1":          float(np.std(fold_f1s)),
        "mean_auc":        float(np.nanmean(fold_aucs)),
        "std_auc":         float(np.nanstd(fold_aucs)),
    }


# =============================================================================
# HYPERPARAMETER TUNING  (Optuna — fold-0, lightweight)
# =============================================================================

def run_optuna(X, y, cat_cols, cont_cols, folds, accelerator="cpu"):
    """
    Tune FT-Transformer architecture kwargs with Optuna on fold-0.
    Each trial trains for up to FT_MAX_EPOCHS with early stopping,
    so runtime is bounded.
    """
    safe_print(f"\n  Running Optuna ({OPTUNA_TRIALS} trials, fold-0 split) ...")

    train_idx, val_idx = folds[0]
    y_arr = y.values

    tr_df, val_df, _ = _make_fold_dfs(
        X.iloc[train_idx], X.iloc[val_idx],
        y_arr[train_idx],  y_arr[val_idx],
        cat_cols, cont_cols,
    )
    # X_va_notar / y_iva: the inner-val split (10 % of fold-0 training).
    # Do NOT use y_arr[val_idx] here — that is the fold-0 TEST set, which
    # has a different length (~20 % of N) and would cause a ValueError.
    X_va_notar = val_df.drop(columns=["target"])
    y_iva      = val_df["target"].values        # inner-val labels — same rows as X_va_notar

    def objective(trial):
        kw = {
            "input_embed_dim": trial.suggest_categorical("input_embed_dim", [16, 32, 64]),
            "num_heads":       trial.suggest_categorical("num_heads", [2, 4, 8]),
            "num_attn_blocks": trial.suggest_int("num_attn_blocks", 2, 6),
            "attn_dropout":    trial.suggest_float("attn_dropout", 0.0, 0.4),
            "ff_dropout":      trial.suggest_float("ff_dropout", 0.0, 0.4),
            "learning_rate":   trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True),
        }
        try:
            model = _build_ft_model(cat_cols, cont_cols, kw, accelerator)
            model.fit(train=tr_df, validation=val_df)
            preds = _ft_predict(model, X_va_notar)
            score = f1_score(y_iva, preds, zero_division=0)
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return score
        except Exception as exc:
            safe_print(f"    Trial failed: {exc}")
            return 0.0

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
    parser = argparse.ArgumentParser(description="FT-Transformer training pipeline")
    parser.add_argument("--tune",  action="store_true",
                        help="Run Optuna hyperparameter tuning")
    parser.add_argument("--no-cv", action="store_true",
                        help="Skip cross-validation step")
    parser.add_argument("--gpu",   action="store_true",
                        help="Use GPU accelerator")
    args        = parser.parse_args()
    accelerator = "gpu" if args.gpu else "cpu"

    safe_print("=" * 65)
    safe_print(f"  {MODEL_NAME} — Independent Training Pipeline  [{accelerator.upper()}]")
    safe_print("=" * 65)

    # -- 1. Load data & folds --------------------------------------------------
    X, y, cat_cols, cont_cols = engineer_features()
    folds   = get_or_create_folds(X, y)
    results = {"model": MODEL_NAME, "accelerator": accelerator}

    # -- 2. Cross-validation ---------------------------------------------------
    if not args.no_cv:
        safe_print(f"\n[Step 1]  5-Fold Stratified CV  (default params)")
        cv = run_cv(X, y, cat_cols, cont_cols, folds, accelerator=accelerator)
        results["cv"] = cv
        safe_print(f"\n  Mean Accuracy : {cv['mean_accuracy']:.4f} +/- {cv['std_accuracy']:.4f}")
        safe_print(f"  Mean F1-score : {cv['mean_f1']:.4f} +/- {cv['std_f1']:.4f}")
        safe_print(f"  Mean ROC-AUC  : {cv['mean_auc']:.4f} +/- {cv['std_auc']:.4f}")

    # -- 3. Optuna tuning (optional) -------------------------------------------
    best_params = None
    if args.tune:
        safe_print("\n[Step 2]  Optuna Hyperparameter Tuning")
        best_params, best_val_f1 = run_optuna(
            X, y, cat_cols, cont_cols, folds, accelerator,
        )
        results["best_params"] = best_params
        results["best_val_f1"] = best_val_f1

        safe_print("\n[Step 2b] CV with tuned params")
        tuned_cv = run_cv(
            X, y, cat_cols, cont_cols, folds,
            model_kwargs=best_params, accelerator=accelerator,
        )
        results["tuned_cv"] = tuned_cv
        safe_print(f"\n  Tuned Mean F1 : {tuned_cv['mean_f1']:.4f} +/- {tuned_cv['std_f1']:.4f}")

    # -- 4. Save results -------------------------------------------------------
    save_results(results, FT_RESULTS_PATH)

    safe_print(f"\n{'=' * 65}")
    safe_print(f"  {MODEL_NAME} complete.")
    safe_print(f"  Results  -> {FT_RESULTS_PATH}")
    safe_print(f"{'=' * 65}")
