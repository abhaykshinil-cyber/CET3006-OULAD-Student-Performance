# =============================================================================
# run_all.py — Master orchestrator: runs all models independently, then compares
# =============================================================================
#
# Usage:
#   python run_all.py                          # default CV for all models
#   python run_all.py --tune                   # add Optuna tuning
#   python run_all.py --skip tabnet ft         # skip TabNet and FT-Transformer
#   python run_all.py --gpu                    # GPU for deep models
#   python run_all.py --no-compare             # skip the comparison step
#
# Each model is launched as a subprocess so failures are isolated —
# one failing model does not prevent the others from running.

import argparse
import os
import subprocess
import sys
import time

def safe_print(*args, **kwargs):
    """Drop-in replacement for print() that never raises UnicodeEncodeError."""
    safe_args = [
        str(a).encode("ascii", errors="replace").decode()
        for a in args
    ]
    print(*safe_args, **kwargs)


SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Ordered model keys -> script filenames
SCRIPTS = {
    "rf":     "train_random_forest.py",
    "xgb":    "train_xgboost.py",
    "tabnet": "train_tabnet.py",
    "ft":     "train_ft_transformer.py",
}

COMPARE_SCRIPT = "compare_models.py"

# Deep models that support the --gpu flag
GPU_CAPABLE = {"tabnet", "ft"}


# =============================================================================
# SUBPROCESS RUNNER
# =============================================================================

def run_script(script_name, extra_args=None):
    """
    Launch script_name as a child process using the current Python interpreter.
    Returns True on success (exit code 0), False otherwise.
    """
    cmd = [sys.executable, os.path.join(SRC_DIR, script_name)]
    if extra_args:
        cmd.extend(extra_args)

    safe_print(f"\n{'-' * 65}")
    safe_print(f"  >  {script_name}  {' '.join(extra_args or [])}")
    safe_print(f"{'-' * 65}")

    t0     = time.perf_counter()
    result = subprocess.run(cmd, cwd=SRC_DIR)
    elapsed = time.perf_counter() - t0

    if result.returncode == 0:
        safe_print(f"\n  [OK]  {script_name}  completed in {elapsed:.1f}s")
        return True
    else:
        safe_print(f"\n  [FAIL]  {script_name}  FAILED  "
              f"(exit code {result.returncode}, {elapsed:.1f}s)")
        safe_print("     Other models are unaffected — continuing ...")
        return False


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="OULAD Research Pipeline — run all models then compare.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Pass --tune to all model scripts (enables Optuna tuning)",
    )
    parser.add_argument(
        "--skip", nargs="*", default=[],
        choices=list(SCRIPTS.keys()),
        metavar="MODEL",
        help="Model keys to skip: rf, xgb, tabnet, ft",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Pass --gpu to TabNet and FT-Transformer",
    )
    parser.add_argument(
        "--no-compare", action="store_true",
        help="Skip the final compare_models step",
    )
    args = parser.parse_args()

    safe_print("=" * 65)
    safe_print("  OULAD RESEARCH PIPELINE — Master Runner")
    safe_print("=" * 65)
    safe_print(f"  Tune mode  : {'ON  (--tune passed to each model)' if args.tune else 'OFF'}")
    safe_print(f"  GPU mode   : {'ON  (--gpu passed to deep models)' if args.gpu  else 'OFF'}")
    safe_print(f"  Skipping   : {', '.join(args.skip) if args.skip else 'none'}")
    safe_print("")

    wall_start = time.perf_counter()
    summary    = {}

    # -- Run each model script independently -----------------------------------
    for key, script in SCRIPTS.items():
        if key in args.skip:
            safe_print(f"  —  Skipping {script}  (--skip {key})")
            summary[key] = "skipped"
            continue

        extra = []
        if args.tune:
            extra.append("--tune")
        if args.gpu and key in GPU_CAPABLE:
            extra.append("--gpu")

        ok = run_script(script, extra)
        summary[key] = "ok" if ok else "FAILED"

    # -- Comparison step --------------------------------------------------------
    if not args.no_compare:
        ok = run_script(COMPARE_SCRIPT)
        summary["compare"] = "ok" if ok else "FAILED"

    # -- Final summary ----------------------------------------------------------
    total = time.perf_counter() - wall_start
    safe_print(f"\n{'=' * 65}")
    safe_print(f"  PIPELINE COMPLETE   (total wall time: {total / 60:.1f} min)")
    safe_print(f"{'=' * 65}")
    icons = {"ok": "[OK]", "skipped": "—", "FAILED": "[FAIL]"}
    for step, status in summary.items():
        icon = icons.get(status, "?")
        safe_print(f"  {icon}  {step:<18}  {status}")
    safe_print("")

    # Exit non-zero if any step failed (useful for CI)
    if any(s == "FAILED" for s in summary.values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
