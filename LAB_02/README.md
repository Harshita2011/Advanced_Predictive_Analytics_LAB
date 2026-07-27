# MDI3003 Lab 02 — Disease Classification Using Decision Trees & Random Forests

**Registration No & Name:** 23MID0043  Harshita Bogieneni
**Course:** MDI3003 — Advanced Predictive Modelling for Medical Data  
**Lab Topic:** Binary disease classification with Decision Trees, Pruned CART, and Random Forests.

---

## 1. Project Overview

This repository compares four classifiers on two medical datasets:

| Dataset | Source | Target |
|---|---|---|
| **Breast Cancer Wisconsin** | `sklearn.datasets.load_breast_cancer` | Malignant (1) vs Benign (0) |
| **Heart Disease** | UCI ML Repository (`ucimlrepo.fetch_ucirepo(id=45)`) | Heart Disease (1) vs Healthy (0) |

For each dataset the pipeline runs:
1. Exploratory Data Analysis (EDA) — distributions, correlations, missing values.
2. Baseline — stratified `DummyClassifier`.
3. Basic CART — unpruned `DecisionTreeClassifier`.
4. Tuned CART — `GridSearchCV` over depth, split/leaf sizes, and `ccp_alpha`.
5. Random Forest — 200-tree ensemble.
6. Evaluation — confusion matrices, ROC/PR curves, calibration plots, feature importance, and error analysis.

All models, plots, and CSV summaries are saved to `models/` and `results/`.

---

## 2. Repository Structure

```
LAB_02/
├── 23MID0043_Lab02_Report.pdf     
├── notebook/                         # Jupyter notebooks and Python scripts
│   ├── 23MID0043_Lab02_BreastCancer_DecisionTree.ipynb
│   ├── 23MID0043_Lab02_HeartDisease_DecisionTree.ipynb
│   ├── 23MID0043_Lab02_BreastCancer_DecisionTree.py      
│   └── 23MID0043_Lab02_HeartDisease_DecisionTree.py      
├── models/                           # Saved joblib model artifacts
├── results/                          # CSVs, plots, and evaluation outputs
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

---

## 3. Reports & Documents

| File | Description |
|---|---|
| `23MID0043_Lab02_Report.docx` | Final lab report with methodology, results, visualisations, and discussion. |
| `FALLSEM2026-27_VL_MDI3003_00100_ELA_2026-07-20_Medical-Diagnosis-Support_-Disease-Classification-Using-Decision-Trees.pdf` | Assignment brief / evaluation rubric for the lab. |

---

## 4. Requirements

- Python 3.10+
- `numpy`
- `pandas`
- `matplotlib`
- `seaborn`
- `scikit-learn`
- `joblib`
- `ucimlrepo` (only for the Heart Disease dataset)

---

## 5. Setup

### Option A — Use your existing Python environment

Install dependencies with pip:

```bash
pip install -r requirements.txt
```

### Option B — Create a fresh virtual environment (recommended)

```bash
# Create the environment
python -m venv venv

# Activate it
# On Windows (Git Bash / WSL):
source venv/Scripts/activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## 6. Running the Project

### 6.1 Run the Jupyter notebooks

```bash
jupyter notebook notebook/23MID0043_Lab02_BreastCancer_DecisionTree.ipynb
jupyter notebook notebook/23MID0043_Lab02_HeartDisease_DecisionTree.ipynb
```

Run all cells (`Cell → Run All`). Outputs are written to `results/` and `models/` automatically.

### 6.2 Run the plain Python scripts

If you prefer not to use Jupyter, run the auto-generated `.py` files:

```bash
# Breast Cancer pipeline
python notebook/23MID0043_Lab02_BreastCancer_DecisionTree.py

# Heart Disease pipeline
python notebook/23MID0043_Lab02_HeartDisease_DecisionTree.py
```

> **Note:** The generated `.py` files mirror the notebook code cells exactly. Some evaluation cells reference a `models_dict` variable that is not defined in the notebooks/scripts. To run the full Python script end-to-end, add the following block **after** the Random Forest cell and **before** the evaluation cells:
>
> ```python
> models_dict = {
>     "Baseline (Dummy)": (dummy_preds, dummy_probs),
>     "Basic Decision Tree": (cart_model.predict(X_test), cart_model.predict_proba(X_test)[:, 1]),
>     "Tuned & Pruned Tree": (tuned_preds, tuned_probs),
>     "Random Forest": (rf_preds, rf_probs)
> }
> colors_list = ["#3172af", "#d95f02", "#2ca02c", "#9467bd"]
> ```

### 6.3 Regenerate `.py` files from notebooks

```bash
python convert_ipynb_to_py.py
```

This re-creates the `.py` versions inside `notebook/` from the current `.ipynb` files.

---

## 7. Pipeline Details

### 7.1 Data Preparation

| Step | Breast Cancer | Heart Disease |
|---|---|---|
| Load | `load_breast_cancer(as_frame=True)` | `fetch_ucirepo(id=45)` |
| Target encoding | `0 → malignant (1)`, `1 → benign (0)` | `num > 0 → disease (1)` |
| Missing values | None | Filled with column median |
| Train/Test split | 80/20 stratified | 80/20 stratified |
| Random seed | `RANDOM_STATE = 42` | `RANDOM_STATE = 42` |

### 7.2 Models

| Model | Key Settings |
|---|---|
| Baseline | `DummyClassifier(strategy='stratified')` |
| Basic CART | `DecisionTreeClassifier(class_weight='balanced')` |
| Tuned CART | `GridSearchCV` with `criterion`, `max_depth=[3,4,5,6]`, `min_samples_split=[2,5,10]`, `min_samples_leaf=[1,2,4]`, `ccp_alpha=[0.0,0.005,0.01,0.02]` |
| Random Forest | `RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_split=5, min_samples_leaf=2, class_weight='balanced', oob_score=True)` |

### 7.3 Evaluation Metrics

- Accuracy
- Recall (Sensitivity)
- Precision
- F1-Score
- ROC-AUC
- Brier Loss
- Out-of-Bag (OOB) score for Random Forest

---

## 8. Generated Outputs

### Models (`models/`)

| File | Description |
|---|---|
| `breast_cancer_baseline.joblib` | Stratified dummy classifier |
| `breast_cancer_basic_cart.joblib` | Unpruned decision tree |
| `breast_cancer_tuned_cart.joblib` | Best tree from `GridSearchCV` |
| `breast_cancer_random_forest.joblib` | 200-tree Random Forest |
| `breast_cancer_best_model.joblib` | Final selected model |
| `23MID0043_Lab02_BreastCancer_Model.joblib` | Submission-named final model |
| `23MID0043_Lab02_Model.joblib` | Generic submission model |
| `heart_disease_*.joblib` | Equivalent Heart Disease artifacts |

### Results (`results/`)

| File | Description |
|---|---|
| `*_eda_class_distribution.png` | Bar + pie charts of target classes |
| `*_eda_feature_distribution.png` | Feature histograms by class |
| `*_eda_correlation_analysis.png` | Correlation heatmap |
| `*_basic_cart_tree.png` | First 3 levels of the basic tree |
| `*_tuned_pruned_tree.png` | Final pruned tree visualization |
| `*_cv_results.csv` | Full `GridSearchCV` results |
| `*_eval_confusion_matrices.png` | 2×2 confusion matrix grid |
| `*_eval_roc_curves.png` | ROC curves for all models |
| `*_eval_pr_curves.png` | Precision–Recall curves |
| `*_eval_calibration_plots.png` | Probability calibration plots |
| `*_feature_importance.csv/png` | Gini and permutation importance |
| `*_model_comparison.csv` | Summary metrics table |
| `*_error_analysis.csv/png` | Test-set error breakdown |

---

---

## 9. Quick Reference Commands

```bash
# Setup
pip install -r requirements.txt

# Convert notebooks to Python
python convert_ipynb_to_py.py

# Run pipelines
python notebook/23MID0043_Lab02_BreastCancer_DecisionTree.py
python notebook/23MID0043_Lab02_HeartDisease_DecisionTree.py

# Or open notebooks
jupyter notebook notebook/
```

---
