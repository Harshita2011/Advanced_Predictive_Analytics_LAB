# LAB 05: Tweet Sentiment Analysis & Comparative Modeling

**Course:** Advanced Predictive Analytics Lab  
**Registration Number:** 23MID0043  

---

## 📌 Overview

This laboratory experiment explores end-to-end sentiment classification on social media data using two distinct benchmark datasets. We implement, evaluate, and benchmark a complete hierarchy of natural language processing (NLP) models—ranging from rule-based and classical TF-IDF machine learning models to deep neural architectures (BiLSTM) and fine-tuned Transformer-based language models (Twitter-RoBERTa).

---

## 📊 Datasets

### 1. Dataset 1 — CardiffNLP TweetEval (`sentiment` subset)
* **Source:** HuggingFace `cardiffnlp/tweet_eval`
* **Labels:** 3 classes — `negative` (0), `neutral` (1), `positive` (2)
* **Data Splits:** Official train (45,615), validation (2,000), and test (12,284) splits.

### 2. Dataset 2 — US Airline Twitter Sentiment
* **Source:** `Tweets.csv`
* **Labels:** 3 classes — `negative`, `neutral`, `positive`
* **Metadata & Entities:** Mentions across 6 US airlines (`Virgin America`, `United`, `Southwest`, `Delta`, `US Airways`, `American`).

---

## 🛠️ Modeling Pipeline & Architectures

The project evaluates models across multiple paradigms:

1. **Baselines & Lexicon Models:**
   * **Dummy Classifier:** Majority-class baseline.
   * **VADER Sentiment:** Rule-based lexicon intensity analyzer.

2. **Classical Machine Learning (TF-IDF):**
   * **Multinomial Naive Bayes (MultinomialNB)**
   * **Logistic Regression** (L2 regularization)
   * **Linear Support Vector Classifier (LinearSVC)**
   * *Feature Representations:* Word n-grams (1-2) and Character n-grams (2-5).

3. **Deep Learning (Recurrent Neural Networks):**
   * **Bidirectional LSTM (BiLSTM):**
     * Trainable Embedding Layer (dim=128, vocab=15k–20k)
     * Bidirectional LSTM layer (64 units)
     * Dropout & Dense classification layers
     * Optimized with Adam and EarlyStopping.

4. **Transfer Learning / Transformer Models:**
   * **Twitter-RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`):**
     * Pre-trained on ~124M tweet corpus.
     * Fine-tuned with Hugging Face `Trainer` API, FP16 precision, and Macro F1 metric optimization.

---

## 📈 Benchmark Results

### Dataset 1: CardiffNLP TweetEval
| Model | Representation | Macro F1 | Weighted F1 | Accuracy | CV Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Word TF-IDF (1-2 gram) | 0.5610 | 0.5759 | 59.30% | 0.6071 |
| **LinearSVC** | Word TF-IDF (1-2 gram) | 0.5684 | 0.5799 | 58.50% | 0.6210 |
| **Char TF-IDF + LinearSVC** | Char TF-IDF (2-5 gram) | 0.5900 | 0.6025 | 60.72% | 0.6386 |
| **BiLSTM** | Word Embedding + BiLSTM | 0.5850 | 0.5947 | 59.48% | — |
| **Twitter-RoBERTa** | Transformer (125M params) | **0.7188** | **0.7166** | **71.74%** | — |

---

### Dataset 2: US Airline Twitter Sentiment
| Model | Representation | Macro F1 | Weighted F1 | Accuracy | CV Macro F1 |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression** | Word TF-IDF (1-2 gram) | 0.7122 | 0.7843 | 79.83% | 0.7124 |
| **LinearSVC** | Word TF-IDF (1-2 gram) | 0.7468 | 0.8056 | 81.28% | 0.7367 |
| **Char TF-IDF + LinearSVC** | Char TF-IDF (2-5 gram) | 0.7457 | 0.8062 | 81.24% | 0.7523 |
| **BiLSTM** | Word Embedding + BiLSTM | 0.7510 | 0.8073 | 80.97% | — |
| **Twitter-RoBERTa** | Transformer (125M params) | **0.8157** | **0.8602** | **85.93%** | — |

---

## 🔬 Key Experiments & Analyses

* **Preprocessing Ablation:** Evaluated minimal cleaning (preserving hashtags, mentions tokenized, emojis) vs. aggressive cleaning (lowercasing, stopword removal, stripping punctuation). Minimal cleaning preserved crucial sentiment signals.
* **Adversarial & Challenge Sets:** Tested robustness against sarcasm, complex negations, mixed sentiments, and emoji-only expressions.
* **Cross-Dataset Domain Transfer:** Evaluated out-of-domain generalization by testing TweetEval models on Airline tweets and vice-versa.
* **Confidence & Abstention Analysis:** Measured accuracy gains when allowing models to abstain on predictions below given probability thresholds.
* **Error Analysis:** Detailed confusion matrix analysis and false positive/negative inspection.

---

## 📂 Repository Structure

```
LAB_05/
├── Dataset1_Tweet_Sentiment.py       # Python script for Dataset 1 pipeline
├── Dataset1_Tweet_Sentiment.ipynb    # Jupyter Notebook for Dataset 1
├── Dataset2_Tweet_Sentiment.py       # Python script for Dataset 2 pipeline
├── Dataset2_Tweet_Sentiment.ipynb    # Jupyter Notebook for Dataset 2
├── Tweets.csv                        # US Airline Sentiment dataset
├── requirements.txt                  # Python package dependencies
├── requirement.txt                   # Dependency requirements alias
├── 23MID0043_Report.pdf              # Comprehensive lab report
├── outputs_dataset1/                 # Generated outputs for Dataset 1
│   ├── figures/                      # Plots, confusion matrices & curves
│   ├── CSV/                          # Metrics, predictions & test reports
│   ├── models/                       # Checkpoints and weights
│   └── metrics/                      # JSON reproducibility summaries
└── outputs_dataset2/                 # Generated outputs for Dataset 2
    ├── figures/                      # Plots, airline distributions & curves
    ├── CSV/                          # Metrics, predictions & test reports
    ├── models/                       # Checkpoints and weights
    └── metrics/                      # JSON reproducibility summaries
```

---

## 🚀 Getting Started

### 1. Installation

Clone the repository and install the dependencies:

```bash
cd LAB_05
pip install -r requirements.txt
```

### 2. Running the Models

#### Run Dataset 1 (CardiffNLP TweetEval):
```bash
python Dataset1_Tweet_Sentiment.py
```

#### Run Dataset 2 (US Airline Sentiment):
```bash
python Dataset2_Tweet_Sentiment.py
```

*All metrics, comparison tables, and figures will be automatically saved to `outputs_dataset1/` and `outputs_dataset2/`.*
