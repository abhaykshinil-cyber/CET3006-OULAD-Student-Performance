import os
import subprocess

# Create logs folder
os.makedirs("logs", exist_ok=True)

# Scripts list (script path, log file, model name)
scripts = [
    ("src/train_random_forest.py", "logs/random_forest_log.txt", "Random Forest"),
    ("src/train_xgboost.py", "logs/xgboost_log.txt", "XGBoost"),
    ("src/train_tabnet.py", "logs/tabnet_log.txt", "TabNet"),
    ("src/train_ft_transformer.py", "logs/ft_transformer_log.txt", "FT-Transformer"),
    ("src/compare_models.py", "logs/comparison_log.txt", "Comparison")
]

# Run scripts and save logs
for script, log_file, model_name in scripts:
    print(f"Running {model_name}...")

    with open(log_file, "w") as f:
        subprocess.run(
            ["python", script],
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True
        )

    print(f"{model_name} completed. Log saved at {log_file}")