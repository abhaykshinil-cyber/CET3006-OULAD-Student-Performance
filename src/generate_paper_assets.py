import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"
FIGURES = RESULTS / "charts"
TABLES = RESULTS

COLORS = {
    "Random Forest": "#355C7D",
    "XGBoost": "#C06C84",
    "TabNet": "#6C9A8B",
    "FT-Transformer": "#F67280",
    "CPU": "#355C7D",
    "GPU": "#F8B195",
}


def load_json(name):
    with open(RESULTS / name, "r", encoding="utf-8") as f:
        return json.load(f)


def save_performance_tables():
    comparison = load_json("comparison_results.json")
    comp_df = pd.DataFrame(comparison["comparison_table"])
    stat_df = pd.DataFrame(comparison["statistical_tests"])
    comp_df.to_csv(TABLES / "table_1_model_performance.csv", index=False)
    stat_df.to_csv(TABLES / "table_2_statistical_tests.csv", index=False)

    xgb = load_json("xgboost_results.json")
    rf = load_json("random_forest_results.json")
    pd.DataFrame(xgb["shap_summary"][:10]).to_csv(
        TABLES / "table_3_top10_shap_features.csv", index=False
    )
    pd.DataFrame(rf["feature_importance"][:10]).to_csv(
        TABLES / "table_4_top10_rf_importance.csv", index=False
    )


def chart_cv_metrics():
    comparison = load_json("comparison_results.json")
    df = pd.DataFrame(comparison["comparison_table"]).copy()
    for col in ["Accuracy", "F1-Score", "ROC-AUC"]:
        df[col] = df[col].astype(float)

    models = df["Model"].tolist()
    x = np.arange(len(models))
    width = 0.22

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width, df["Accuracy"], width, label="Accuracy", color="#355C7D")
    ax.bar(x, df["F1-Score"], width, label="F1-score", color="#6C9A8B")
    ax.bar(x + width, df["ROC-AUC"], width, label="ROC-AUC", color="#C06C84")

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0.88, 1.0)
    ax.set_title("Cross-validation performance comparison")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False, ncols=3, loc="upper center")

    for offset, col in [(-width, "Accuracy"), (0, "F1-Score"), (width, "ROC-AUC")]:
        for i, value in enumerate(df[col]):
            ax.text(x[i] + offset, value + 0.002, f"{value:.4f}",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    fig.savefig(FIGURES / "figure_1_cv_metric_comparison.png", dpi=220)
    plt.close(fig)


def chart_fold_distribution():
    paths = {
        "Random Forest": load_json("random_forest_results.json")["cv"]["fold_f1s"],
        "XGBoost": load_json("xgboost_results.json")["cv"]["fold_f1s"],
        "TabNet": load_json("tabnet_results.json")["cv"]["fold_f1s"],
        "FT-Transformer": load_json("ft_transformer_results.json")["cv"]["fold_f1s"],
    }

    fig, ax = plt.subplots(figsize=(9, 5.2))
    data = [paths[k] for k in paths]
    labels = list(paths.keys())
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    for patch, label in zip(bp["boxes"], labels):
        patch.set_facecolor(COLORS[label])
        patch.set_alpha(0.65)
    ax.set_xticklabels(labels, rotation=15, ha="right")
    ax.set_ylabel("Fold-level F1-score")
    ax.set_title("Distribution of F1-scores across 5 folds")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    plt.tight_layout()
    fig.savefig(FIGURES / "figure_2_fold_f1_distribution.png", dpi=220)
    plt.close(fig)


def chart_feature_importance():
    rf = pd.DataFrame(load_json("random_forest_results.json")["feature_importance"][:8])
    xgb = pd.DataFrame(load_json("xgboost_results.json")["feature_importance"][:8])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.8))

    axes[0].barh(rf["feature"][::-1], rf["importance"][::-1], color="#355C7D")
    axes[0].set_title("Random Forest top features")
    axes[0].set_xlabel("Importance")
    axes[0].grid(axis="x", linestyle="--", alpha=0.3)

    axes[1].barh(xgb["feature"][::-1], xgb["importance"][::-1], color="#C06C84")
    axes[1].set_title("XGBoost top features")
    axes[1].set_xlabel("Importance")
    axes[1].grid(axis="x", linestyle="--", alpha=0.3)

    fig.suptitle("Most influential features in the tree-based models", y=1.02)
    plt.tight_layout()
    fig.savefig(FIGURES / "figure_3_feature_importance_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def chart_shap_summary():
    shap_df = pd.read_csv(RESULTS / "shap_summary.csv").head(10)
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    ax.barh(shap_df["feature"][::-1], shap_df["mean_abs_shap"][::-1], color="#F67280")
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_title("Top 10 SHAP features from the XGBoost model")
    ax.grid(axis="x", linestyle="--", alpha=0.3)
    plt.tight_layout()
    fig.savefig(FIGURES / "figure_4_shap_summary.png", dpi=220)
    plt.close(fig)


def _clean_numeric(series):
    return pd.to_numeric(series.astype(str).str.replace("[^0-9.]", "", regex=True), errors="coerce")


def chart_cpu_gpu():
    gpu_df = pd.read_csv(RESULTS / "oulad_gpu_comparison.csv")
    gpu_df["Train Time GPU (s)"] = _clean_numeric(gpu_df["Train Time GPU (s)"])
    gpu_df["Infer Time GPU (s)"] = _clean_numeric(gpu_df["Infer Time GPU (s)"])
    gpu_df = gpu_df.dropna(subset=["Train Time GPU (s)", "Infer Time GPU (s)"])

    x = np.arange(len(gpu_df))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    axes[0].bar(x - width / 2, gpu_df["Train Time CPU (s)"], width, label="CPU", color=COLORS["CPU"])
    axes[0].bar(x + width / 2, gpu_df["Train Time GPU (s)"], width, label="GPU", color=COLORS["GPU"])
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(gpu_df["Model"])
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Training time")
    axes[0].grid(axis="y", linestyle="--", alpha=0.3)

    axes[1].bar(x - width / 2, gpu_df["Infer Time CPU (s)"], width, label="CPU", color=COLORS["CPU"])
    axes[1].bar(x + width / 2, gpu_df["Infer Time GPU (s)"], width, label="GPU", color=COLORS["GPU"])
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(gpu_df["Model"])
    axes[1].set_ylabel("Seconds")
    axes[1].set_title("Inference time")
    axes[1].grid(axis="y", linestyle="--", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=2, frameon=False)
    fig.suptitle("CPU versus GPU execution time for the deep-learning models", y=1.03)
    plt.tight_layout()
    fig.savefig(FIGURES / "figure_5_cpu_gpu_times.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def draw_methodology_diagram():
    fig, ax = plt.subplots(figsize=(12, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    boxes = [
        (0.4, 2.0, 2.0, 1.0, "#DCEAF7", "OULAD source files\nstudentInfo, studentVle,\nstudentAssessment, assessments"),
        (2.8, 2.0, 2.0, 1.0, "#E6F4EA", "Feature engineering\nVLE and assessment\naggregations"),
        (5.2, 2.0, 2.0, 1.0, "#FFF4D6", "Leakage-safe\npreprocessing\nwithin each fold"),
        (7.6, 2.0, 2.0, 1.0, "#FCE4EC", "Shared 5-fold\nstratified CV\nsplits"),
        (10.0, 2.0, 1.6, 1.0, "#E8EAF6", "Model training\nand evaluation"),
    ]

    for x, y, w, h, c, text in boxes:
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.1, edgecolor="#444444", facecolor=c
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    for start, end in [(2.4, 2.8), (4.8, 5.2), (7.2, 7.6), (9.6, 10.0)]:
        ax.annotate("", xy=(end, 2.5), xytext=(start, 2.5),
                    arrowprops=dict(arrowstyle="->", lw=1.6, color="#555555"))

    ax.text(10.8, 1.0, "Random Forest\nXGBoost\nTabNet\nFT-Transformer",
            ha="center", va="center", fontsize=9)
    ax.text(10.8, 3.9, "Accuracy\nF1-score\nROC-AUC\nTiming / Interpretation",
            ha="center", va="center", fontsize=9)
    ax.annotate("", xy=(10.8, 3.0), xytext=(10.8, 3.55),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#555555"))
    ax.annotate("", xy=(10.8, 2.0), xytext=(10.8, 1.45),
                arrowprops=dict(arrowstyle="->", lw=1.4, color="#555555"))

    ax.set_title("Methodology pipeline used in the experiment", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(FIGURES / "figure_6_methodology_pipeline.png", dpi=220)
    plt.close(fig)


def draw_system_diagram():
    fig, ax = plt.subplots(figsize=(11.2, 6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    nodes = [
        (0.7, 5.4, 2.0, 0.9, "#DCEAF7", "Raw OULAD data"),
        (0.7, 3.8, 2.0, 0.9, "#DCEAF7", "Data loader"),
        (3.2, 4.6, 2.1, 0.9, "#E6F4EA", "Feature builder"),
        (5.8, 4.6, 2.1, 0.9, "#FFF4D6", "Fold manager"),
        (8.3, 5.4, 2.0, 0.9, "#FCE4EC", "Tree models"),
        (8.3, 3.8, 2.0, 0.9, "#F3E5F5", "Deep models"),
        (8.3, 2.2, 2.0, 0.9, "#E8EAF6", "Comparison engine"),
        (3.2, 2.2, 2.1, 0.9, "#FBE9E7", "Result store"),
    ]

    for x, y, w, h, c, text in nodes:
        rect = patches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.08",
            linewidth=1.1, edgecolor="#444444", facecolor=c
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    arrows = [
        ((2.7, 5.85), (3.2, 5.05)),
        ((2.7, 4.25), (3.2, 5.0)),
        ((5.3, 5.05), (5.8, 5.05)),
        ((7.9, 5.05), (8.3, 5.85)),
        ((7.9, 5.05), (8.3, 4.25)),
        ((9.3, 5.4), (9.3, 3.1)),
        ((9.3, 3.8), (9.3, 3.1)),
        ((8.3, 2.65), (5.3, 2.65)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start,
                    arrowprops=dict(arrowstyle="->", lw=1.5, color="#555555"))

    ax.text(6.7, 6.15, "Shared preprocessing and saved CV splits", fontsize=9, ha="center")
    ax.text(4.25, 1.5, "CSV / JSON tables, charts, and summaries", fontsize=9, ha="center")
    ax.set_title("System architecture for the research pipeline", fontsize=13, pad=12)
    plt.tight_layout()
    fig.savefig(FIGURES / "figure_7_system_architecture.png", dpi=220)
    plt.close(fig)


def main():
    save_performance_tables()
    chart_cv_metrics()
    chart_fold_distribution()
    chart_feature_importance()
    chart_shap_summary()
    chart_cpu_gpu()
    draw_methodology_diagram()
    draw_system_diagram()
    print(f"Saved figures to {FIGURES}")
    print(f"Saved tables to {TABLES}")


if __name__ == "__main__":
    main()
