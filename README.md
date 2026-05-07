# Student Performance Prediction on OULAD

This repository contains the code and experiment outputs for a CET3006 research project on student performance prediction using the Open University Learning Analytics Dataset (OULAD).

The current project compares four tabular-learning models:

- `Random Forest`
- `XGBoost`
- `TabNet`
- `FT-Transformer`

The study focuses on:

- predictive performance on binary student outcome prediction
- interpretability through feature importance and SHAP analysis
- computational trade-offs between classical tree models and deep tabular models
- paper-ready charts, tables, and diagrams for the final research report

---

## Project Summary

The pipeline uses OULAD student records together with engineered engagement and assessment features to predict:

- positive class: `Pass` and `Distinction`
- negative class: `Fail` and `Withdrawn`

The main comparison uses shared 5-fold stratified cross-validation and reports:

- `Accuracy`
- `F1-score`
- `ROC-AUC`

The repository also includes:

- statistical significance testing
- Random Forest and XGBoost feature importance
- SHAP-based interpretability for XGBoost
- CPU and GPU timing comparison assets

---

## Repository Structure

```text
.
|-- data/
|   `-- oulad/              # dataset CSVs (excluded from repo — see Dataset section)
|-- diagrams/
|   `-- old/               # archived early diagrams
|-- results/
|   |-- charts/
|   |   |-- diagram1_feature_engineering.png
|   |   |-- diagram2_methodology_pipeline.png
|   |   |-- diagram3_system_architecture.png
|   |   |-- figure_1_cv_metric_comparison.png
|   |   |-- figure_2_fold_f1_distribution.png
|   |   |-- figure_3_feature_importance_comparison.png
|   |   |-- figure_4_shap_summary.png
|   |   |-- figure_5_cpu_gpu_times.png
|   |   |-- figure_8_class_distribution.png
|   |   |-- figure_9_radar_comparison.png
|   |   `-- figure_10_statistical_significance.png
|   |-- folds/
|   |   `-- fold_indices.json
|   |-- comparison_results.json
|   |-- comparison_table.csv
|   |-- feature_importance_rf.csv
|   |-- feature_importance_xgb.csv
|   |-- ft_transformer_results.json
|   |-- insight_summary.txt
|   |-- oulad_cpu_results.csv
|   |-- oulad_gpu_comparison.csv
|   |-- random_forest_results.json
|   |-- shap_summary.csv
|   |-- statistical_tests.csv
|   |-- table_1_model_performance.csv
|   |-- table_2_statistical_tests.csv
|   |-- table_3_top10_shap_features.csv
|   |-- table_4_top10_rf_importance.csv
|   |-- tabnet_results.json
|   `-- xgboost_results.json
|-- src/
|   |-- build_submission_doc.ps1
|   |-- compare_models.py
|   |-- config.py
|   |-- data_loader.py
|   |-- generate_paper_assets.py
|   |-- run_all.py
|   |-- train_ft_transformer.py
|   |-- train_random_forest.py
|   |-- train_tabnet.py
|   |-- train_xgboost.py
|   `-- visualize_all.py
|-- table screenshots/
|   `-- old/               # archived table screenshots
|-- vizualized_charts/
|   `-- old/               # archived early charts
|-- requirements.txt
|-- run_with_logs.py
`-- README.md
```

---

## Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**.

- Official dataset source: [Open University Learning Analytics Dataset](https://analyse.kmi.open.ac.uk/open_dataset)
- Dataset paper: Kuzilek, Hlosta and Zdrahal (2017), *Scientific Data*

If the CSV files are not present, place them under:

```text
data/oulad/
```

---

## Methods

### Models

- `Random Forest`
  - Implemented with `sklearn.ensemble.RandomForestClassifier`
  - Used as a strong classical ensemble baseline

- `XGBoost`
  - Implemented with `xgboost.XGBClassifier`
  - Used for boosted-tree comparison and SHAP analysis

- `TabNet`
  - Implemented with `pytorch-tabnet`
  - Evaluated as a deep tabular learning model

- `FT-Transformer`
  - Implemented with `pytorch-tabular`
  - Evaluated as a Transformer-based tabular model

### Feature Engineering

The pipeline builds a shared tabular feature set from:

- `studentInfo.csv`
- `studentVle.csv`
- `studentAssessment.csv`
- `assessments.csv`

Key engineered features include:

- `total_clicks`
- `early_clicks`
- `late_clicks`
- `num_active_days`
- `unique_activities`
- `avg_daily_clicks`
- `avg_score`
- `max_score`
- `num_submissions`

### Evaluation Design

- binary target: `Pass/Distinction -> 1`, `Fail/Withdrawn -> 0`
- shared 5-fold stratified cross-validation
- leakage-safe preprocessing inside each fold
- evaluation metrics: `Accuracy`, `F1-score`, `ROC-AUC`
- statistical testing on fold-level F1 scores

---

## Main Output Files

### Model Results

- `results/random_forest_results.json`
- `results/xgboost_results.json`
- `results/tabnet_results.json`
- `results/ft_transformer_results.json`

### Comparison Results

- `results/comparison_results.json`
- `results/comparison_table.csv`
- `results/statistical_tests.csv`
- `results/insight_summary.txt`

### Interpretability Results

- `results/feature_importance_rf.csv`
- `results/feature_importance_xgb.csv`
- `results/shap_summary.csv`
- `results/shap_values.npy`

### Paper Assets (Charts & Diagrams)

- `results/charts/diagram1_feature_engineering.png`
- `results/charts/diagram2_methodology_pipeline.png`
- `results/charts/diagram3_system_architecture.png`
- `results/charts/figure_1_cv_metric_comparison.png`
- `results/charts/figure_2_fold_f1_distribution.png`
- `results/charts/figure_3_feature_importance_comparison.png`
- `results/charts/figure_4_shap_summary.png`
- `results/charts/figure_5_cpu_gpu_times.png`
- `results/charts/figure_8_class_distribution.png`
- `results/charts/figure_9_radar_comparison.png`
- `results/charts/figure_10_statistical_significance.png`

### Generated Tables for the Paper

- `results/table_1_model_performance.csv`
- `results/table_2_statistical_tests.csv`
- `results/table_3_top10_shap_features.csv`
- `results/table_4_top10_rf_importance.csv`

---

## Setup

Create and activate the environment:

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

Typical dependencies used by this project include:

- `pandas`
- `numpy`
- `scikit-learn`
- `xgboost`
- `torch`
- `pytorch-tabnet`
- `pytorch-tabular`
- `matplotlib`
- `scipy`
- `shap`
- `optuna`

---

## Usage

### Run all model experiments

```bash
cd src
python run_all.py
```

Optional examples:

```bash
python run_all.py --tune
python run_all.py --gpu
python run_all.py --skip tabnet ft
```

### Run individual models

```bash
python train_random_forest.py
python train_xgboost.py
python train_tabnet.py
python train_ft_transformer.py
```

### Generate comparison outputs

```bash
python compare_models.py
```

### Generate paper assets

```bash
python generate_paper_assets.py
```

### Build the Word submission document

This step uses Microsoft Word automation on Windows:

```powershell
.\src\build_submission_doc.ps1
```

---

## Academic Integrity

This repository accompanies a university assessment.

If you are a student:

- do not submit this work as your own
- follow your institution's academic integrity policy
- cite all external sources correctly

---

## Citation

If you refer to this repository, cite the project in a form similar to:

> Abhay Kalathil Shinil (2026). *Accuracy, Interpretability, and Efficiency in Student Performance Prediction on OULAD: A Comparative Study of Random Forest, XGBoost, TabNet, and FT-Transformer*. CET3006 Research Project, University of Sunderland.

---

## License

Add a project license if you plan to share or reuse this repository outside the current academic context.
