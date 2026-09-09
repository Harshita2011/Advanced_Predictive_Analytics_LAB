# Lab 07: Two-Stage Recommender System (Random Forest Re-ranking)

**Course:** Advanced Predictive Analytics (MDI3003)  
**Author / Register No.:** 23MID0043  

---

## 📌 Repository Overview & Multi-Dataset Structure

This repository contains the complete experimental suite, machine learning pipelines, trained models, visual figures, and evaluation artifacts across **three distinct datasets**:

```
LAB_07/
├── Dataset_1_Initial_OnlineRetail/     # [Dataset 1] Initial Online Retail subset benchmark
│   ├── 23MID0043_Lab07_Recommender_RF.ipynb
│   ├── models/ (Trained Random Forest model)
│   ├── figures/ (11 Exploratory & Diagnostic visual plots)
│   ├── outputs_csv/ (Evaluation, Ablation, Sensitivity, Uncertainty metrics)
│   └── artifacts/ (Data cards, Split manifests, Feature schemas)
│
├── Dataset_2_UCI_OnlineRetail_541k/    # [Dataset 2] Full UCI Online Retail Dataset (541,909 rows)
│   ├── run_lab07_pipeline.py           # End-to-end reproducible pipeline script
│   ├── 23MID0043_Lab07_Recommender_RF.ipynb
│   ├── 23MID0043_Lab07_Report.pdf & .docx
│   ├── models/ (Trained Random Forest model)
│   ├── figures/ (Full set of 11 high-res visualization figures)
│   ├── outputs_csv/ (Ranking metrics, ablation, uncertainty tables)
│   └── artifacts/ (dataset_card.json, split_manifest.json, feature_schema.json)
│
├── Dataset_3_Retailrocket/             # [Dataset 3] Retailrocket E-commerce Clickstream & Events
│   ├── 23MID0043_Lab07_Retailrocket_Recommender.ipynb
│   ├── models/ (Trained Retailrocket RF recommender model)
│   ├── figures/ (12 PNG conversion funnel, category & ranking plots)
│   └── outputs_csv/ (Behavioral recommendations & uncertainty metrics)
│
├── 23MID0043_Lab07_Report.pdf          # 16-Page Comprehensive Lab Report (PDF)
├── 23MID0043_Lab07_Report.docx         # 16-Page Comprehensive Lab Report (Word)
├── 23MID0043_Lab07_All_3_Datasets_Results.zip # Complete compressed benchmark archive
└── run_lab07_pipeline.py               # Main Python pipeline script
```

---

## 🗂️ Dataset Breakdown

### 1️⃣ Dataset 1: Initial Online Retail Dataset (`Dataset_1_Initial_OnlineRetail`)
- **Description:** Initial exploratory run on an Online Retail transaction subset for baseline model prototyping and candidate generation validation.
- **Key Artifacts Included:**
  - **Notebook:** `23MID0043_Lab07_Recommender_RF.ipynb`
  - **Trained Model:** `models/random_forest.joblib`
  - **Evaluation Metrics (CSV):** `outputs_csv/` (Ranking metrics, candidate recall, feature ablation, error analysis, negative sampling sensitivity, efficiency table, recommendations)
  - **Figures (11 PNGs):** `figures/` (Transaction volume, top items, customer frequency, popularity tail, class balance, feature importance, precision/recall @ K, coverage, system recall)
  - **Metadata:** `artifacts/` (dataset_card.json, split_manifest.json, feature_schema.json, candidate_policy.json, environment_manifest.json)

---

### 2️⃣ Dataset 2: UCI Online Retail Dataset (`Dataset_2_UCI_OnlineRetail_541k`)
- **Description:** Complete benchmark on the **full UCI Machine Learning Repository Online Retail Dataset** containing **541,909 raw transaction records** (spanning 01/12/2010 to 09/12/2011).
- **Pipeline Highlights:**
  - Strict temporal 80/20 train/test split preventing data leakage.
  - Multi-source candidate retrieval (Popularity + Item-Item Cosine Similarity + User Purchase History) generating Top-$N$ candidates.
  - 14 engineered features spanning RFM (Recency, Frequency, Monetary), item velocity, co-purchase synergy, and user-item interactions.
  - Calibrated Random Forest Classifier for Top-$K$ re-ranking.
- **Key Artifacts Included:**
  - **Production Script:** `run_lab07_pipeline.py`
  - **Jupyter Notebook:** `23MID0043_Lab07_Recommender_RF.ipynb`
  - **Trained Model:** `models/random_forest.joblib`
  - **Lab Reports:** `23MID0043_Lab07_Report.pdf` and `23MID0043_Lab07_Report.docx`
  - **Evaluation Metrics (CSV):**
    - `23MID0043_Lab07_Ranking_Metrics.csv` (Precision@K, Recall@K, NDCG@K, MAP@K, MRR@K, Catalog Coverage, Gini Diversity)
    - `23MID0043_Lab07_Candidate_Recall.csv`
    - `23MID0043_Lab07_Feature_Ablation.csv`
    - `23MID0043_Lab07_Error_Analysis.csv`
    - `23MID0043_Lab07_Negative_Sampling_Sensitivity.csv`
    - `23MID0043_Lab07_Efficiency_Table.csv`
    - `23MID0043_Lab07_Recommendations.csv`
    - `23MID0043_Lab07_Advanced_Uncertainty.csv`
  - **Figures & Visualizations (PNG):** Complete high-resolution exploratory and ranking diagnostic plots.

---

### 3️⃣ Dataset 3: Retailrocket E-commerce Dataset (`Dataset_3_Retailrocket`)
- **Description:** Large-scale implicit feedback clickstream dataset containing visitor interactions (`view`, `addtocart`, `transaction`), item property metadata, and category trees.
- **Pipeline Highlights:**
  - Implicit behavioral funnel modeling with weighted positive interaction scoring.
  - Candidate retrieval handling sparse multi-session visitor traffic.
  - Random Forest point-wise re-ranking on session-level behavioral signals.
- **Key Artifacts Included:**
  - **Jupyter Notebook:** `23MID0043_Lab07_Retailrocket_Recommender.ipynb`
  - **Trained Model:** `models/random_forest_retailrocket.joblib`
  - **Evaluation Metrics (CSV):**
    - `23MID0043_Lab07_Retailrocket_Ranking_Metrics.csv`
    - `23MID0043_Lab07_Retailrocket_Candidate_Recall.csv`
    - `23MID0043_Lab07_Retailrocket_Feature_Ablation.csv`
    - `23MID0043_Lab07_Retailrocket_Error_Analysis.csv`
    - `23MID0043_Lab07_Retailrocket_Negative_Sampling_Sensitivity.csv`
    - `23MID0043_Lab07_Retailrocket_Efficiency_Table.csv`
    - `23MID0043_Lab07_Retailrocket_Recommendations.csv`
    - `23MID0043_Lab07_Retailrocket_Advanced_Uncertainty.csv`
  - **Figures (12 PNGs):** Event type distribution, daily event volume, visitor activity, conversion funnel, top categories, item properties, class balance, feature importance, precision/recall curves, and catalog coverage.

---

## 📊 Summary of Comparative Benchmark Performance

| Dataset | Total Records | Candidates Retreived | Candidate Pool Recall@100 | Re-Ranker Precision@10 | Re-Ranker Recall@10 | Re-Ranker NDCG@10 | Catalog Coverage |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dataset 1 (Initial Retail)** | ~50k rows | Top-100 | 88.4% | 0.0482 | 0.2814 | 0.3120 | 41.2% |
| **Dataset 2 (UCI 541k Rows)** | 541,909 rows | Top-100 | 92.6% | 0.0512 | 0.3045 | 0.3418 | 44.8% |
| **Dataset 3 (Retailrocket)** | 2,756,101 events | Top-100 | 86.1% | 0.0395 | 0.2450 | 0.2780 | 38.6% |

---

## 🚀 How to Run the Pipelines

### 1. Run UCI Dataset 2 Full Pipeline:
```bash
python run_lab07_pipeline.py
```

### 2. Open and Run Jupyter Notebooks:
```bash
jupyter notebook Dataset_1_Initial_OnlineRetail/23MID0043_Lab07_Recommender_RF.ipynb
jupyter notebook Dataset_2_UCI_OnlineRetail_541k/23MID0043_Lab07_Recommender_RF.ipynb
jupyter notebook Dataset_3_Retailrocket/23MID0043_Lab07_Retailrocket_Recommender.ipynb
```
