# Student Performance Prediction on OULAD

This repository contains the code, experiment outputs, paper assets, and submission documents for a CET3006 research project on student performance prediction using the Open University Learning Analytics Dataset (OULAD).

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
- CET3006-aligned research paper drafts and Word submission files

---

## Repository Structure

```text
.
|-- data/
|   `-- oulad/
|       |-- assessments.csv
|       |-- courses.csv
|       |-- studentAssessment.csv
|       |-- studentInfo.csv
|       |-- studentRegistration.csv
|       |-- studentVle.csv
|       `-- vle.csv
|-- docs/
|   |-- CET3006_research_paper_aligned.md
|   |-- CET3006_final_submission.docx
|   |-- CET3006_final_submission_resaved.docx
|   |-- CET3006_submission_ready.docx
|   |-- CET3006_RESEARCH_PAPER_ABHAY_KALATHIL_SHINIL.pdf
|   `-- research_paper_draft.md
|-- results/
|   |-- charts/
|   |   |-- figure_1_cv_metric_comparison.png
|   |   |-- figure_2_fold_f1_distribution.png
|   |   |-- figure_3_feature_importance_comparison.png
|   |   |-- figure_4_shap_summary.png
|   |   |-- figure_5_cpu_gpu_times.png
|   |   |-- figure_6_methodology_pipeline.png
|   |   `-- figure_7_system_architecture.png
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
|   `-- train_xgboost.py
|-- diagrams/
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

### Paper Assets

- `results/charts/figure_1_cv_metric_comparison.png`
- `results/charts/figure_2_fold_f1_distribution.png`
- `results/charts/figure_3_feature_importance_comparison.png`
- `results/charts/figure_4_shap_summary.png`
- `results/charts/figure_5_cpu_gpu_times.png`
- `results/charts/figure_6_methodology_pipeline.png`
- `results/charts/figure_7_system_architecture.png`

### Generated Tables for the Paper

- `results/table_1_model_performance.csv`
- `results/table_2_statistical_tests.csv`
- `results/table_3_top10_shap_features.csv`
- `results/table_4_top10_rf_importance.csv`

### Paper and Submission Files

- `docs/CET3006_research_paper_aligned.md`
- `docs/research_paper_draft.md`
- `docs/CET3006_final_submission.docx`
- `docs/CET3006_final_submission_resaved.docx`
- `docs/CET3006_RESEARCH_PAPER_ABHAY_KALATHIL_SHINIL.pdf`

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

## Research Paper Notes

The CET3006-aligned paper currently includes:

- a structured abstract
- contributions section
- literature review with 16 references
- methodology, experiments, results, conclusion, and future work
- embedded chart and diagram placeholders in the markdown draft
- Word submission exports

The main aligned draft is:

- `docs/CET3006_research_paper_aligned.md`

The safer resaved Word version is:

- `docs/CET3006_final_submission_resaved.docx`

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
