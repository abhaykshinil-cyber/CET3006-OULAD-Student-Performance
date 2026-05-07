import pandas as pd
import matplotlib.pyplot as plt

# =========================
# LOAD DATA
# =========================
cpu_df = pd.read_csv("oulad_cpu_results.csv")
gpu_df = pd.read_csv("oulad_gpu_comparison.csv")

print("CPU Results:\n", cpu_df.head())
print("\nGPU Comparison:\n", gpu_df.head())

# =========================
# CPU MODEL COMPARISON
# =========================

# Accuracy
plt.figure()
plt.bar(cpu_df["Model"], cpu_df["Accuracy"])
plt.title("Accuracy Comparison (CPU)")
plt.xlabel("Models")
plt.ylabel("Accuracy")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("cpu_accuracy.png")
plt.show()

# F1 Score
plt.figure()
plt.bar(cpu_df["Model"], cpu_df["F1"])
plt.title("F1 Score Comparison (CPU)")
plt.xlabel("Models")
plt.ylabel("F1 Score")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("cpu_f1.png")
plt.show()

# Training Time
plt.figure()
plt.bar(cpu_df["Model"], cpu_df["Training Time (CPU, s)"])
plt.title("Training Time (CPU)")
plt.xlabel("Models")
plt.ylabel("Seconds")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("cpu_training_time.png")
plt.show()

# Inference Time
plt.figure()
plt.bar(cpu_df["Model"], cpu_df["Inference Time (CPU, s)"])
plt.title("Inference Time (CPU)")
plt.xlabel("Models")
plt.ylabel("Seconds")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig("cpu_inference_time.png")
plt.show()

# =========================
# CPU vs GPU COMPARISON (FIXED)
# =========================

import numpy as np

# Remove rows where GPU data is missing (like Random Forest)
gpu_clean = gpu_df.dropna(subset=["Train Time GPU (s)", "Infer Time GPU (s)"])

models = gpu_clean["Model"]
x = np.arange(len(models))
width = 0.35

# -------------------------
# Training Time Comparison
# -------------------------
plt.figure()
plt.bar(x - width/2, gpu_clean["Train Time CPU (s)"], width, label="CPU")
plt.bar(x + width/2, gpu_clean["Train Time GPU (s)"], width, label="GPU")

plt.xticks(x, models, rotation=30)
plt.title("Training Time: CPU vs GPU")
plt.ylabel("Seconds")
plt.legend()
plt.tight_layout()
plt.savefig("cpu_vs_gpu_training_fixed.png")
plt.show()

# -------------------------
# Inference Time Comparison
# -------------------------
plt.figure()
plt.bar(x - width/2, gpu_clean["Infer Time CPU (s)"], width, label="CPU")
plt.bar(x + width/2, gpu_clean["Infer Time GPU (s)"], width, label="GPU")

plt.xticks(x, models, rotation=30)
plt.title("Inference Time: CPU vs GPU")
plt.ylabel("Seconds")
plt.legend()
plt.tight_layout()
plt.savefig("cpu_vs_gpu_inference_fixed.png")
plt.show()