# =============================================================================
# OULAD Tabular Model Comparison — CPU vs GPU Experimental Design
# =============================================================================
#
# RESEARCH GOAL:
#   Evaluate performance vs computational cost under resource-constrained
#   environments where GPU acceleration may not be available.
#
# EXPERIMENT 1 — MAIN (CPU only)
#   All four models run exclusively on CPU.
#   Metrics: Accuracy, F1, Training Time (CPU), Inference Time (CPU)
#   Models:  Random Forest | TabNet | FT-Transformer
#
# EXPERIMENT 2 — SECONDARY (GPU comparison)
#   Deep-learning models re-run on GPU to quantify acceleration.
#   Skipped automatically when no CUDA device is found.
#   Metrics: Training Time (GPU), Inference Time (GPU), Speedup factor
#   Models:  TabNet (GPU) | FT-Transformer (GPU)
#            [Random Forest remains on CPU]
#
# INSTALL (Google Colab):
#   !pip install pytorch-tabnet pytorch-tabular --quiet
#
# DATA FILES (update paths below if needed):
#   studentInfo-6923c1cf.csv
#   studentVle-b701c576.csv
# =============================================================================

import gc
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from pytorch_tabnet.tab_model import TabNetClassifier
from pytorch_tabular import TabularModel
from pytorch_tabular.config import DataConfig, OptimizerConfig, TrainerConfig

# pytorch-tabular relocated FTTransformerConfig between major versions
try:
    from pytorch_tabular.models.ft_transformer.config import FTTransformerConfig
except ImportError:                                          # pytorch-tabular ≥ 1.0
    from pytorch_tabular.models import FTTransformerConfig


# =============================================================================
# CONFIGURATION — update these paths to match your file locations
# =============================================================================
# Running locally:  place files in the same folder as this script, or set
#                   full paths, e.g. "/content/drive/MyDrive/oulad/..."
# Running in Colab: upload the files and set paths to "/content/..."

STUDENT_INFO_PATH = "../data/oulad/studentInfo.csv"
STUDENT_VLE_PATH  = "../data/oulad/studentVle.csv"
# =============================================================================


# ─────────────────────────────────────────────────────────────────────────────
# ENVIRONMENT DETECTION
# ─────────────────────────────────────────────────────────────────────────────

CUDA_AVAILABLE = torch.cuda.is_available()

print("=" * 70)
print("ENVIRONMENT")
print("=" * 70)
print(f"PyTorch version   : {torch.__version__}")
print(f"CUDA available    : {CUDA_AVAILABLE}")
if CUDA_AVAILABLE:
    print(f"GPU device        : {torch.cuda.get_device_name(0)}")
    total_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU memory        : {total_mem:.1f} GB")
else:
    print("GPU device        : none — Experiment 2 will be skipped")
print()


# =============================================================================
# SECTION 1 — DATA LOADING & PREPROCESSING
# =============================================================================

def load_oulad(info_path: str = STUDENT_INFO_PATH,
               vle_path:  str = STUDENT_VLE_PATH):
    """
    Load OULAD, aggregate VLE clicks, create binary target, label-encode.

    Parameters
    ----------
    info_path : str
        Path to the studentInfo CSV file.
    vle_path : str
        Path to the studentVle CSV file.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns integer-encoded).
    y : pd.Series
        Binary target (Pass/Distinction=1, Fail/Withdrawn=0).
    cat_cols : list[str]
        Names of originally-categorical (now integer-encoded) columns.
    """
    print(f"Loading: {info_path}")
    print(f"Loading: {vle_path}")
    info_df = pd.read_csv(info_path)
    vle_df  = pd.read_csv(vle_path)

    # Sum total VLE interactions per student × module × presentation
    vle_grouped = (
        vle_df.groupby(["id_student", "code_module", "code_presentation"])["sum_click"]
        .sum()
        .reset_index()
    )

    df = pd.merge(
        info_df, vle_grouped,
        on=["id_student", "code_module", "code_presentation"],
        how="left",
    )
    df["sum_click"] = df["sum_click"].fillna(0)

    # Binary outcome: success (Pass / Distinction) vs failure (Fail / Withdrawn)
    df["final_result"] = df["final_result"].map(
        {"Pass": 1, "Distinction": 1, "Fail": 0, "Withdrawn": 0}
    )
    df = df.dropna(subset=["final_result"])
    df = df.drop(columns=["id_student", "code_module", "code_presentation"])

    cat_cols = df.select_dtypes(include=["object"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    X = df.drop(columns=["final_result"])
    y = df["final_result"].astype(int)
    return X, y, cat_cols


# =============================================================================
# SECTION 2 — TRAIN / TEST SPLIT
# =============================================================================

X, y, categorical_cols = load_oulad(STUDENT_INFO_PATH, STUDENT_VLE_PATH)
feature_cols    = X.columns.tolist()
continuous_cols = [c for c in feature_cols if c not in categorical_cols]

# Stratified 80 / 20 — maintain both DataFrame views (FT-Transformer) and
# numpy views (Random Forest, TabNet).
X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train  = X_train_df.values
X_test   = X_test_df.values
y_train  = y_train_s.values
y_test   = y_test_s.values

print("=" * 70)
print("DATASET SUMMARY")
print("=" * 70)
print(f"Total rows        : {X.shape[0]}")
print(f"Features          : {X.shape[1]}")
print(f"Train / Test      : {X_train.shape[0]} / {X_test.shape[0]}")
print(f"Categorical cols  : {categorical_cols}")
print(f"Continuous  cols  : {continuous_cols}")
print(f"Class balance     : {y_train.mean():.3f} positive (train set)")
print()


# =============================================================================
# SECTION 3 — SHARED HELPERS
# =============================================================================

def cuda_sync():
    """Synchronise CUDA stream so wall-clock time reflects true GPU work."""
    if CUDA_AVAILABLE:
        torch.cuda.synchronize()


def timed_fit(model, X, y, fit_kwargs=None, use_cuda_sync=False):
    """
    Fit model, return elapsed seconds.
    use_cuda_sync=True adds CUDA synchronisation for accurate GPU timing.
    """
    if use_cuda_sync:
        cuda_sync()
    t0 = time.perf_counter()
    if fit_kwargs:
        model.fit(X, y, **fit_kwargs)
    else:
        model.fit(X, y)
    if use_cuda_sync:
        cuda_sync()
    return time.perf_counter() - t0


def timed_predict(predict_fn, X, use_cuda_sync=False):
    """
    Run predict_fn(X), return (predictions, elapsed_seconds).
    use_cuda_sync=True adds CUDA synchronisation for accurate GPU timing.
    """
    if use_cuda_sync:
        cuda_sync()
    t0 = time.perf_counter()
    preds = predict_fn(X)
    if use_cuda_sync:
        cuda_sync()
    return preds, time.perf_counter() - t0


def free_gpu_memory():
    """Release unused GPU memory between experiments."""
    gc.collect()
    if CUDA_AVAILABLE:
        torch.cuda.empty_cache()


# ── FT-Transformer DataFrames (shared between both experiments) ───────────────

def build_ft_dataframes():
    """Return (train_df, val_df, test_df) with correct dtypes for pytorch-tabular."""
    tr = X_train_df.copy()
    tr["target"] = y_train
    te = X_test_df.copy()
    te["target"] = y_test

    # Feature dtypes
    for col in categorical_cols:
        tr[col] = tr[col].astype(int)
        te[col] = te[col].astype(int)
    for col in continuous_cols:
        tr[col] = tr[col].astype(float)
        te[col] = te[col].astype(float)

    # KEEP target as integer (pytorch-tabular will treat it as class label)
    tr["target"] = tr["target"].astype("int64")
    te["target"] = te["target"].astype("int64")

    tr_split, val_split = train_test_split(
        tr, test_size=0.1, random_state=42, stratify=tr["target"]
    )
    return tr_split, val_split, te


# ── Factory: TabNet ───────────────────────────────────────────────────────────

def build_tabnet(device: str) -> TabNetClassifier:
    """
    Instantiate a TabNetClassifier pinned to the requested device.
    device: "cpu" | "cuda"
    TabNet uses device_name= (not torch.device) to set computation target.
    """
    return TabNetClassifier(
        n_d=32,           # decision-step embedding width
        n_a=32,           # attention-step embedding width (mirrors n_d)
        n_steps=5,        # sequential attention steps
        gamma=1.3,        # feature-reuse coefficient
        n_independent=2,
        n_shared=2,
        device_name=device,   # ← explicit CPU / CUDA assignment
        seed=42,
        verbose=0,
    )


# ── Factory: FT-Transformer ───────────────────────────────────────────────────

# Shared architecture config (device-independent)
_FT_MODEL_CONFIG = FTTransformerConfig(
    task="classification",
    input_embed_dim=32,     # embedding dimension per feature token
    num_heads=4,            # multi-head attention heads
    num_attn_blocks=4,      # transformer encoder blocks
    attn_dropout=0.1,
    ff_dropout=0.1,
    learning_rate=1e-3,
)

_FT_DATA_CONFIG = DataConfig(
    target=["target"],
    continuous_cols=continuous_cols,
    categorical_cols=categorical_cols,
    num_workers=0,  # safer on Windows
)


def build_ft_trainer(accelerator: str) -> TrainerConfig:
    """
    Build a TrainerConfig for the requested accelerator.
    accelerator: "cpu" | "gpu"

    Handles both pytorch-tabular ≥ 1.0 (trainer_kwargs-based) and
    legacy < 1.0 (gpus= keyword).
    """
    try:
        kw = {"accelerator": accelerator, "enable_model_summary": False}
        if accelerator == "gpu":
            kw["devices"] = 1                 # use one GPU
        return TrainerConfig(
            max_epochs=20,
            batch_size=512,
            early_stopping="valid_loss",
            early_stopping_patience=5,
            checkpoints=None,                 # no checkpoint files
            load_best=True,
            progress_bar="none",
            trainer_kwargs=kw,
        )
    except TypeError:
        # Legacy pytorch-tabular < 1.0
        gpus = 0 if accelerator == "cpu" else 1
        return TrainerConfig(max_epochs=20, batch_size=512,
                             gpus=gpus, early_stopping=True)


def build_ft_model(accelerator: str) -> TabularModel:
    """Return a fresh TabularModel pinned to the requested accelerator."""
    return TabularModel(
        data_config=_FT_DATA_CONFIG,
        model_config=_FT_MODEL_CONFIG,
        trainer_config=build_ft_trainer(accelerator),
        optimizer_config=OptimizerConfig(),
    )


def ft_predict(ft_model, X_df_no_target) -> np.ndarray:
    out = ft_model.predict(X_df_no_target)
    # Use the class label column
    return out["target_prediction"].astype(int).values


# ─────────────────────────────────────────────────────────────────────────────
# Results storage
# ─────────────────────────────────────────────────────────────────────────────

cpu_results = {}   # {model_name: {"acc", "f1", "train_t", "inf_t"}}
gpu_results = {}   # {model_name: {"train_t", "inf_t"}}


# =============================================================================
# EXPERIMENT 1 — ALL MODELS ON CPU (main experiment)
# =============================================================================

print("=" * 70)
print("EXPERIMENT 1 — CPU  (main results)")
print("Models evaluated under resource-constrained environment (no GPU).")
print("=" * 70)


# ── 1a. Random Forest ────────────────────────────────────────────────────────

print("\n[1/4] Random Forest")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)

t0 = time.perf_counter()
rf.fit(X_train, y_train)
rf_train_t = time.perf_counter() - t0

rf_preds, rf_inf_t = timed_predict(rf.predict, X_test)

rf_acc = accuracy_score(y_test, rf_preds)
rf_f1  = f1_score(y_test, rf_preds)
cpu_results["Random Forest"] = dict(acc=rf_acc, f1=rf_f1,
                                    train_t=rf_train_t, inf_t=rf_inf_t)
print(f"  Accuracy         : {rf_acc:.4f}")
print(f"  F1-score         : {rf_f1:.4f}")
print(f"  Training time    : {rf_train_t:.3f} s")
print(f"  Inference time   : {rf_inf_t:.4f} s")


# ── 1b. TabNet — CPU ─────────────────────────────────────────────────────────

print("\n[2/4] TabNet  (device=cpu)")
X_tr_tn = X_train.astype(np.float32)
X_te_tn = X_test.astype(np.float32)

tabnet_cpu = build_tabnet("cpu")

t0 = time.perf_counter()
tabnet_cpu.fit(
    X_train=X_tr_tn, y_train=y_train,
    eval_set=[(X_te_tn, y_test)],
    max_epochs=50,
    patience=10,
    batch_size=1024,
)
tabnet_cpu_train_t = time.perf_counter() - t0

tabnet_preds_cpu, tabnet_cpu_inf_t = timed_predict(tabnet_cpu.predict, X_te_tn)
tabnet_preds_cpu = tabnet_preds_cpu.reshape(-1)

tabnet_acc = accuracy_score(y_test, tabnet_preds_cpu)
tabnet_f1  = f1_score(y_test, tabnet_preds_cpu)
cpu_results["TabNet"] = dict(acc=tabnet_acc, f1=tabnet_f1,
                              train_t=tabnet_cpu_train_t, inf_t=tabnet_cpu_inf_t)
print(f"  Accuracy         : {tabnet_acc:.4f}")
print(f"  F1-score         : {tabnet_f1:.4f}")
print(f"  Training time    : {tabnet_cpu_train_t:.3f} s")
print(f"  Inference time   : {tabnet_cpu_inf_t:.4f} s")

del tabnet_cpu
free_gpu_memory()


# ── 1c. FT-Transformer — CPU ─────────────────────────────────────────────────

print("\n[3/4] FT-Transformer  (accelerator=cpu)")
train_pt, val_pt, test_pt = build_ft_dataframes()
X_test_ft = test_pt.drop(columns=["target"])

ft_cpu = build_ft_model("cpu")

t0 = time.perf_counter()
ft_cpu.fit(train=train_pt, validation=val_pt)
ft_cpu_train_t = time.perf_counter() - t0

ft_cpu.load_best_model()

ft_preds_cpu, ft_cpu_inf_t = timed_predict(
    lambda df: ft_predict(ft_cpu, df), X_test_ft
)

ft_acc = accuracy_score(y_test, ft_preds_cpu)
ft_f1  = f1_score(y_test, ft_preds_cpu)
cpu_results["FT-Transformer"] = dict(acc=ft_acc, f1=ft_f1,
                                      train_t=ft_cpu_train_t, inf_t=ft_cpu_inf_t)
print(f"  Accuracy         : {ft_acc:.4f}")
print(f"  F1-score         : {ft_f1:.4f}")
print(f"  Training time    : {ft_cpu_train_t:.3f} s")
print(f"  Inference time   : {ft_cpu_inf_t:.4f} s")

del ft_cpu
free_gpu_memory()


# ── 1d. TabPFN — CPU ─────────────────────────────────────────────────────────

print("\n[4/4] TabPFN skipped")


# ── Experiment 1 Results Table ───────────────────────────────────────────────

print("\n" + "=" * 70)
print("TABLE 1 — CPU Results (main experiment)")
print("=" * 70)
cpu_df = pd.DataFrame(
    [
        [name,
         f"{r['acc']:.4f}",
         f"{r['f1']:.4f}",
         f"{r['train_t']:.3f}",
         f"{r['inf_t']:.4f}"]
        for name, r in cpu_results.items()
    ],
    columns=["Model", "Accuracy", "F1",
             "Training Time (CPU, s)", "Inference Time (CPU, s)"],
)
print(cpu_df.to_string(index=False))
cpu_df.to_csv("../results/oulad_cpu_results.csv", index=False)
print("\nSaved → ../results/oulad_cpu_results.csv")


# =============================================================================
# EXPERIMENT 2 — GPU COMPARISON (secondary analysis)
# Deep-learning models only: TabNet and FT-Transformer.
# Skipped if no CUDA device is available.
# =============================================================================

print("\n" + "=" * 70)
print("EXPERIMENT 2 — GPU Comparison (secondary / acceleration analysis)")
print("Deep-learning models only (Random Forest remains on CPU).")
print("=" * 70)

if not CUDA_AVAILABLE:
    print("\n  No CUDA device found — Experiment 2 skipped.")
    print("  To run GPU comparison, execute this script on a machine with a")
    print("  compatible NVIDIA GPU and CUDA-enabled PyTorch.\n")
else:
    print(f"\n  Running on: {torch.cuda.get_device_name(0)}\n")

    # ── 2a. TabNet — GPU ─────────────────────────────────────────────────────

    print("[1/2] TabNet  (device=cuda)")
    tabnet_gpu = build_tabnet("cuda")

    t0 = time.perf_counter()
    cuda_sync()                             # flush any pending GPU ops before timing
    tabnet_gpu.fit(
        X_train=X_tr_tn, y_train=y_train,
        eval_set=[(X_te_tn, y_test)],
        max_epochs=50,
        patience=10,
        batch_size=1024,
    )
    cuda_sync()                             # wait for GPU to finish before stopping clock
    tabnet_gpu_train_t = time.perf_counter() - t0

    cuda_sync()
    t1 = time.perf_counter()
    tabnet_preds_gpu = tabnet_gpu.predict(X_te_tn).reshape(-1)
    cuda_sync()
    tabnet_gpu_inf_t = time.perf_counter() - t1

    gpu_results["TabNet"] = dict(train_t=tabnet_gpu_train_t,
                                  inf_t=tabnet_gpu_inf_t)
    print(f"  Training time    : {tabnet_gpu_train_t:.3f} s  "
          f"(CPU was {cpu_results['TabNet']['train_t']:.3f} s, "
          f"speedup {cpu_results['TabNet']['train_t'] / tabnet_gpu_train_t:.1f}×)")
    print(f"  Inference time   : {tabnet_gpu_inf_t:.4f} s  "
          f"(CPU was {cpu_results['TabNet']['inf_t']:.4f} s)")

    del tabnet_gpu
    free_gpu_memory()

    # ── 2b. FT-Transformer — GPU ─────────────────────────────────────────────

    print("\n[2/2] FT-Transformer  (accelerator=gpu)")
    ft_gpu = build_ft_model("gpu")

    cuda_sync()
    t0 = time.perf_counter()
    ft_gpu.fit(train=train_pt, validation=val_pt)
    cuda_sync()
    ft_gpu_train_t = time.perf_counter() - t0

    ft_gpu.load_best_model()

    cuda_sync()
    t1 = time.perf_counter()
    ft_preds_gpu = ft_predict(ft_gpu, X_test_ft)
    cuda_sync()
    ft_gpu_inf_t = time.perf_counter() - t1

    gpu_results["FT-Transformer"] = dict(train_t=ft_gpu_train_t,
                                          inf_t=ft_gpu_inf_t)
    print(f"  Training time    : {ft_gpu_train_t:.3f} s  "
          f"(CPU was {cpu_results['FT-Transformer']['train_t']:.3f} s, "
          f"speedup {cpu_results['FT-Transformer']['train_t'] / ft_gpu_train_t:.1f}×)")
    print(f"  Inference time   : {ft_gpu_inf_t:.4f} s  "
          f"(CPU was {cpu_results['FT-Transformer']['inf_t']:.4f} s)")

    del ft_gpu
    free_gpu_memory()

    # ── Experiment 2 Results Table ───────────────────────────────────────────

    def fmt(val):
        """Format a timing value; return 'N/A' if not available."""
        return f"{val:.3f}" if val is not None else "N/A"

    def speedup(cpu_t, gpu_t):
        return f"{cpu_t / gpu_t:.2f}×" if gpu_t is not None else "N/A"

    all_models = ["Random Forest", "TabNet", "FT-Transformer"]
    gpu_rows = []
    for name in all_models:
        c = cpu_results[name]
        g = gpu_results.get(name)
        gpu_rows.append([
            name,
            fmt(c["train_t"]),
            fmt(g["train_t"]) if g else "—  (CPU only)",
            speedup(c["train_t"], g["train_t"] if g else None),
            fmt(c["inf_t"]),
            fmt(g["inf_t"]) if g else "—  (CPU only)",
            speedup(c["inf_t"], g["inf_t"] if g else None),
        ])

    print("\n" + "=" * 70)
    print("TABLE 2 — CPU vs GPU Comparison (secondary experiment)")
    print("=" * 70)
    gpu_df = pd.DataFrame(
        gpu_rows,
        columns=[
            "Model",
            "Train Time CPU (s)", "Train Time GPU (s)", "Train Speedup",
            "Infer Time CPU (s)", "Infer Time GPU (s)", "Infer Speedup",
        ],
    )
    print(gpu_df.to_string(index=False))
    gpu_df.to_csv("../results/oulad_gpu_comparison.csv", index=False)
    print("\nSaved → ../results/oulad_gpu_comparison.csv")


# =============================================================================
# SECTION 6 — FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 70)
print("FINAL SUMMARY")
print("=" * 70)
print("""
Main finding (Experiment 1):
  All models were evaluated on CPU to reflect real-world deployment in
  resource-constrained environments.  Accuracy and F1-score provide the
  primary performance comparison; training and inference times capture
  the computational cost.

Secondary finding (Experiment 2, GPU only):
  TabNet and FT-Transformer were re-trained on GPU to quantify hardware
  acceleration.  Speedup factors show how much faster each model trains
  with a GPU — relevant for researchers with access to cloud GPUs.

Interpretation:
  • A high Accuracy / Training-Time ratio on CPU indicates efficiency
    suitable for environments without specialised hardware.
  • Large GPU speedup factors signal models that benefit most from
    hardware investment but may be impractical on CPU alone.
""")
print("CSV files saved:")
print("  oulad_cpu_results.csv       ← Experiment 1 (main)")
if CUDA_AVAILABLE:
    print("  oulad_gpu_comparison.csv    ← Experiment 2 (secondary)")
