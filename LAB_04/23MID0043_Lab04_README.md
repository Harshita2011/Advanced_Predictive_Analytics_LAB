# MDI3003 Lab 04 — Customer Segmentation (Naive Bayes)

**Registration Number:** 23MID0043

## Dataset used (core, assessed)

**Dataset A** — JanataHack Customer Segmentation (Analytics Vidhya competition, mirrored on Kaggle as
`vetrirah/customer`). Verified sources:
- https://www.analyticsvidhya.com/datahack/contest/janatahack-customer-segmentation
- https://www.kaggle.com/datasets/vetrirah/customer

File used: `Train_aBjfeNk.csv` (8,068 records, target `Segmentation` ∈ {A, B, C, D}).
SHA-256 checksum is recorded in `artifacts/dataset_card.json`.

`Test_LqhgPWU.csv` from the same Kaggle competition is an **unlabeled** hold-out used for the original
Kaggle leaderboard, not the lab's own locked test set — this notebook creates its own locked 80/20
stratified split from `Train_aBjfeNk.csv`, per the lab manual's instructions.

**Dataset B** (`marketing_campaign.csv`, Kaggle `imakash3011/customer-personality-analysis`) is evaluated in the **Advanced-Learner Research Extension (Appendix B8 & B9)** as a controlled secondary benchmark per Section 7.7 & 16.1 of the manual. It features 2,240 records with campaign response prediction (`Response` ∈ {0, 1}) as the target. It is evaluated independently and not merged with Dataset A.

## Contents

| File / folder | Description |
|---|---|
| `23MID0043_Lab04_CustomerSegmentation.ipynb` | Full executed notebook (runs top-to-bottom, seed=42). Includes dataset load, EDA, leakage-safe preprocessing, 4 core models, 5-fold CV, feature-group ablation, locked-test evaluation, confidence/review policy, error analysis, new-customer prediction, acceptance tests, and research extensions. |
| `23MID0043_Lab04_Report.pdf` | Technical report — dataset governance, label-provenance audit, CV/test results, feature-group ablation, confusion matrices, confidence analysis, error analysis, responsible-analytics discussion, final recommendation. |
| `23MID0043_Lab04_CV_Results.csv` | Mean/SD macro F1, accuracy, weighted F1 per core model (identical 5-fold CV). |
| `23MID0043_Lab04_Test_Results.csv` | One-time locked-test metrics for the selected model. |
| `23MID0043_Lab04_NewCustomer_Predictions.csv` | Sample new-customer predictions with posterior distribution and review status. |
| `23MID0043_Lab04_Error_Analysis.csv` | Interpreted misclassified test cases with business-consequence notes. |
| `figures/` | class_distribution, cv_comparison, feature_group_ablation, confusion_matrix (count + normalized), coverage_error_curve, confidence_distribution, secondary_dataset_cv_comparison. |
| `models/selected_pipeline.joblib` | Fitted, reload-verified scikit-learn pipeline (selected model). |
| `artifacts/` | versions.json, dataset_card.json, label_provenance_audit.json, feature_manifest.json, split_manifest.csv, frozen_review_threshold.json, model_selection_note.txt, classification_report.csv, coverage_error_table.csv, cv_fold_level_macro_f1.csv, feature_group_ablation.csv, data_audit.csv, test_predictions_full.csv, acceptance_tests.json, verified_dataset_comparison.csv, secondary_dataset_cv_results.csv, secondary_dataset_card.json, **plus advanced-learner extension artifacts** (see below). |

## Advanced-learner bonus extensions (Appendix B of the notebook)

All extensions below run on frozen splits (seed=42), so results are directly comparable.

| Extension | Artifact(s) | Headline result |
|---|---|---|
| Verified Dataset Comparison (Table 12) | `artifacts/verified_dataset_comparison.csv` | Complete dataset governance audit comparing Datasets A (JanataHack), B (Customer Personality), C (UCI Online Retail II), and D (UCI Bank Marketing) across records, modalities, target suitability, and licensing. |
| Controlled Secondary Dataset Experiment | `artifacts/secondary_dataset_cv_results.csv`, `secondary_dataset_card.json`, `figures/secondary_dataset_cv_comparison.png` | Evaluated 4 Naive Bayes pipelines on Dataset B (2,240 records, target `Response`). Dummy (macro F1 = 0.849), BernoulliNB (0.849), and CategoricalNB_mixed (0.849) achieve strong performance on the imbalanced campaign response task. |
| ComplementNB (justified sparse non-negative model) | `artifacts/extension_cv_results.csv`, `extension_test_results.csv` | Locked-test macro F1 = 0.4141 — underperforms core CategoricalNB/BernoulliNB, consistent with ComplementNB's text-classification design assumptions being a weak fit for this mildly-imbalanced tabular task (class ratio ≈1.22). |
| Logistic Regression benchmark (non-Naive-Bayes) | same as above | Locked-test macro F1 = 0.5041 — modestly ahead of all core NB models. |
| Bootstrap 95% CI for the selected core model | `artifacts/bootstrap_ci_macro_f1.json`, `figures/bootstrap_macro_f1_ci.png` | CategoricalNB_mixed locked-test macro F1 = 0.4853, 95% CI **[0.4614, 0.5086]** (2,000 resamples) — confirms the small CV gaps between core models are not decisive. |
| Quantitative fairness audit | `artifacts/fairness_audit.csv` | `Gender`: Female macro F1 0.482 (n=690) vs Male 0.485 (n=924) — no material gap. `Ever_Married`: "No" 0.342 (n=676) vs "Yes" 0.390 (n=909) — a gap worth flagging; a 29-row missing-value group is marked insufficient-sample and not interpreted. |
| FT-Transformer vs. core NB baseline | `artifacts/ft_transformer_results.json`, `nb_vs_transformer_comparison.csv`, `figures/ft_transformer_confusion_matrix.png`, `figures/nb_vs_transformer_comparison.png` | Minimal feature-tokenizer Transformer (26,948 params, ~30s CPU training): locked-test macro F1 = 0.5174, weighted F1 = 0.5260, accuracy = 0.5353 — clears the core model's bootstrap CI upper bound, at the cost of materially higher training complexity than CategoricalNB's near-instant fit. Single run, no repeated-seed variance estimate yet. |
| Temporal-drift holdout | `artifacts/extension_not_applicable_notes.json` | **Not applicable** — Dataset A has no timestamp/date field. |
| BERT/DistilBERT text-augmented model | `artifacts/extension_not_applicable_notes.json` | **Not applicable** — Dataset A has no free-text field; the manual disallows converting categorical columns into artificial sentences to justify BERT. |

## Headline results

- **Selected model:** `CategoricalNB_mixed` (mixed-feature representation, alpha=1.0)
- **Selection basis:** mean 5-fold CV macro F1 = 0.4845 (SD 0.0101), ahead of BernoulliNB (0.4801, SD
  0.0087), GaussianNB (0.3719), and well above the Dummy baseline (0.1097). Selected before any access to
  the locked test set.
- **Locked-test macro F1:** 0.4853 (accuracy 0.5118), closely matching the CV estimate — no evidence of
  overfitting during selection.
- **Feature-group ablation:** demographic features alone (macro F1 0.4851) explain almost all of the
  combined model's performance (0.4845); the single available psychographic (`Var_1`) and behavioral
  (`Spending_Score`) proxies add little on their own.
- **Confidence:** no threshold achieved selective error ≤ 15% on validation, so the review threshold was
  frozen at 0.50 (moderate-confidence cutoff); Naive Bayes posteriors here are not well calibrated,
  consistent with the manual's caution against treating them as certainty.

## How to re-run

1. Place `Train_aBjfeNk.csv` in the same folder as the notebook (or edit `DATA_PATH` in cell 2).
2. Run the notebook top-to-bottom with a clean kernel (`pandas`, `numpy`, `scikit-learn`, `matplotlib`,
   `joblib` — versions pinned in `artifacts/versions.json`).
3. All results/figures/artifacts regenerate under `lab04_outputs/`.

## Known limitations (see report Section 8–10 for full discussion)

- `Segmentation`'s true construction rule is undisclosed by the dataset source — this is treated as a
  pedagogical exercise, not a validated real-world segmentation model.
- Model relies heavily on demographic (including sensitive) attributes — Gender, Ever_Married, Age. Any
  production use requires a fairness/policy review before relying on these features.
- Overall accuracy (~51%) reflects a genuinely hard 4-class problem with substantial class overlap, not a
  pipeline defect — this is corroborated by the public Kaggle leaderboard for this dataset, where even
  strong models plateau in a similar accuracy range.
