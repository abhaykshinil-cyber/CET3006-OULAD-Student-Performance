# =============================================================================
# config.py — Shared configuration for the OULAD research pipeline
# =============================================================================
#
# All training scripts import from here so every model uses identical
# paths, seeds, and hyper-parameter budgets.

import os

# ── Directory layout ──────────────────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR    = os.path.join(BASE_DIR, "data", "oulad")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
FOLDS_DIR   = os.path.join(RESULTS_DIR, "folds")
CHARTS_DIR  = os.path.join(RESULTS_DIR, "charts")

# ── Raw data paths ────────────────────────────────────────────────────────────
STUDENT_INFO_PATH       = os.path.join(DATA_DIR, "studentInfo.csv")
STUDENT_VLE_PATH        = os.path.join(DATA_DIR, "studentVle.csv")
STUDENT_ASSESSMENT_PATH = os.path.join(DATA_DIR, "studentAssessment.csv")
ASSESSMENTS_PATH        = os.path.join(DATA_DIR, "assessments.csv")

# ── Persisted fold indices ────────────────────────────────────────────────────
FOLD_INDICES_PATH = os.path.join(FOLDS_DIR, "fold_indices.json")

# ── Per-model result files ────────────────────────────────────────────────────
RF_RESULTS_PATH  = os.path.join(RESULTS_DIR, "random_forest_results.json")
XGB_RESULTS_PATH = os.path.join(RESULTS_DIR, "xgboost_results.json")
TN_RESULTS_PATH  = os.path.join(RESULTS_DIR, "tabnet_results.json")
FT_RESULTS_PATH  = os.path.join(RESULTS_DIR, "ft_transformer_results.json")

SHAP_VALUES_PATH = os.path.join(RESULTS_DIR, "shap_values.npy")
COMPARISON_PATH  = os.path.join(RESULTS_DIR, "comparison_results.json")

# ── Reproducibility ───────────────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Cross-validation ──────────────────────────────────────────────────────────
N_FOLDS = 5

# Inner validation fraction (used inside each fold for deep models)
VAL_FRACTION = 0.10

# ── Optuna budget ─────────────────────────────────────────────────────────────
OPTUNA_TRIALS = 25          # per model; keep lightweight

# ── Deep model epoch limits ───────────────────────────────────────────────────
TABNET_MAX_EPOCHS    = 50
TABNET_PATIENCE      = 10
FT_MAX_EPOCHS        = 20
FT_PATIENCE          = 5

# ── Ensure output directories exist on import ─────────────────────────────────
for _d in (RESULTS_DIR, FOLDS_DIR, CHARTS_DIR):
    os.makedirs(_d, exist_ok=True)

# =============================================================================
# SAFE PRINT  — works on any terminal encoding (Windows cp1252, UTF-8, etc.)
# =============================================================================

import sys as _sys

def safe_print(*args, **kwargs):
    """
    Drop-in replacement for print() that never raises UnicodeEncodeError.

    Accepts all standard print() arguments: end, sep, flush, file.
    Each positional arg is encoded to ASCII with "?" substitution for any
    character the terminal cannot represent (e.g. Windows cp1252 cmd.exe).
    """
    safe_args = [
        str(a).encode("ascii", errors="replace").decode()
        for a in args
    ]
    print(*safe_args, **kwargs)

