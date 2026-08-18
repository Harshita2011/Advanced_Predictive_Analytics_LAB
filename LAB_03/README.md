# MDI3003 Lab 03 — Benchmark-Aligned Email Classification & Spam Detection (BiLSTM vs. TF-IDF)

**Registration No & Name:** 23MID0043 Harshita Bogineni  
**Course:** MDI3003 — Advanced Predictive Analytics / Analytics for Medical Data  
**Lab Topic:** Benchmark-Aligned Email Intent Classification, Enron & SpamAssassin Spam Detection using TF-IDF Baselines, BiLSTM Neural Networks, and LLM Reply Generation.  

---

## 1. Project Overview

This repository implements a comparative study between classical TF-IDF machine learning pipelines (Multinomial Naive Bayes, Complement Naive Bayes, Logistic Regression, LinearSVC) and Deep Learning Sequential architectures (Bidirectional LSTM with learned embeddings) across two standard benchmark email spam datasets and business intent classification:

1. **Enron Spam Dataset** (~33,000 emails) — Real-world enterprise emails categorized into Spam (1) vs. Ham (0).
2. **SpamAssassin Dataset** (~6,000 emails) — Public benchmark email corpus for evaluating spam filter robustness.
3. **Email Business Intent** — Categorization of email intents and auto-drafting contextual response emails using LLMs (OpenAI API).

The pipeline covers raw text audit, text normalization/preprocessing, tokenization & sequence padding, TF-IDF feature extraction, 5-fold cross-validation, Deep Learning BiLSTM model training with EarlyStopping, model serialization (`.keras` and `.pkl`), visual diagnostic reporting (Length Distributions, Confusion Matrices, ROC Curves), and automated LLM draft generation.

---

## 2. Repository Structure

```
LAB_03/
├── 23MID0043_Lab03_Report.pdf                            # Comprehensive technical laboratory report
├── 23MID0043_Lab03_Email_Classification_with_BiLSTM.ipynb    # End-to-end template notebook (Colab & local)
├── 23MID0043_Lab03_Email_Classification_with_BiLSTM.py       # Auto-generated Python script for template
├── 23MID0043_Lab03_EnronSpam_Classifier.ipynb                # Enron dataset TF-IDF & BiLSTM pipeline notebook
├── 23MID0043_Lab03_EnronSpam_Classifier.py                    # Auto-generated Python script for Enron pipeline
├── 23MID0043_Lab03_SpamAssassin_Classifier.ipynb             # SpamAssassin dataset TF-IDF & BiLSTM pipeline notebook
├── 23MID0043_Lab03_SpamAssassin_Classifier.py                 # Auto-generated Python script for SpamAssassin pipeline
├── convert_ipynb_to_py.py                                # Utility script to regenerate .py files from .ipynb
├── requirements.txt                                      # Python package dependencies
├── enron_spam_data.csv                                   # Enron Spam raw CSV dataset
├── spam_assassin.csv                                     # SpamAssassin raw CSV dataset
├── enron_bilstm.keras                                    # Saved Keras HDF5 model for Enron BiLSTM
├── spamassassin_bilstm.keras                             # Saved Keras HDF5 model for SpamAssassin BiLSTM
├── enron_best_tfidf.pkl                                  # Serialized TF-IDF vectorizer artifact for Enron
├── spamassassin_best_tfidf.pkl                           # Serialized TF-IDF vectorizer artifact for SpamAssassin
├── eronspamlengthdistribution.png                        # Length distribution diagnostic plot (Enron)
├── eronspamconfusionandroc.png                           # Confusion matrix & ROC Curve grid (Enron)
├── Spam_assassin1.png                                    # Length distribution diagnostic plot (SpamAssassin)
└── spamassasssincm.png                                   # Confusion matrix & ROC Curve grid (SpamAssassin)
```

---

## 3. Environment & Dependencies

- **Python Version:** 3.10+
- **Core Libraries:**
  - `pandas` & `numpy` (data handling & manipulation)
  - `scikit-learn` (TF-IDF vectorization, baselines, metrics, cross-validation)
  - `tensorflow` / `keras` (BiLSTM architecture, Embedding, Tokenizer, pad_sequences)
  - `matplotlib` & `seaborn` (visualization & diagnostic plots)
  - `joblib` (model & vectorizer serialization)
  - `openai` & `python-dotenv` (LLM prompt engineering for email response drafting)

Install all requirements using:

```bash
pip install -r requirements.txt
```

---

## 4. Execution Guide

### Option A — Run via Jupyter Notebooks
Open and execute cells sequentially (`Cell → Run All`):

```bash
jupyter notebook 23MID0043_Lab03_EnronSpam_Classifier.ipynb
jupyter notebook 23MID0043_Lab03_SpamAssassin_Classifier.ipynb
jupyter notebook 23MID0043_Lab03_Email_Classification_with_BiLSTM.ipynb
```

### Option B — Run via Plain Python Scripts
Execute the generated standalone `.py` scripts:

```bash
python 23MID0043_Lab03_EnronSpam_Classifier.py
python 23MID0043_Lab03_SpamAssassin_Classifier.py
python 23MID0043_Lab03_Email_Classification_with_BiLSTM.py
```

### Option C — Re-generate Python Scripts
To update all `.py` files from current `.ipynb` notebooks:

```bash
python convert_ipynb_to_py.py
```

---

## 5. Pipeline Methodology

Each dataset pipeline follows a leakage-safe 12-step methodology:

1. **Environment Setup & Seed Initialization** (`RANDOM_STATE = 42`).
2. **Dataset Audit & Loading**:
   - Cleaning missing values (`subject` & `body`).
   - Text concatenation: `text = "Subject: " + subject + "\nBody: " + body`.
3. **Exploratory Text Analysis**:
   - Subject & Body word count distributions per class (Ham vs. Spam).
4. **Leakage-Safe Train/Test Split**:
   - 80/20 stratified split (`stratify=y`).
5. **TF-IDF Baseline Models**:
   - `TfidfVectorizer(max_features=5000, stop_words='english', ngram_range=(1,2))`.
   - Baseline Classifiers: Dummy (Stratified), Multinomial Naive Bayes, Complement Naive Bayes, Logistic Regression (`C=1.0`), LinearSVC (`C=1.0`).
   - 5-Fold Stratified Cross-Validation on training set.
6. **Deep Learning BiLSTM Architecture**:
   - `Tokenizer(num_words=10000)` & `pad_sequences(maxlen=200, padding='post', truncating='post')`.
   - Sequential Architecture:
     - `Embedding(input_dim=10000, output_dim=64, input_length=200)`
     - `Bidirectional(LSTM(64, return_sequences=False))`
     - `Dropout(0.5)`
     - `Dense(32, activation='relu')`
     - `Dense(1, activation='sigmoid')`
   - Optimization: `Adam(learning_rate=1e-3)` with `binary_crossentropy` loss.
   - EarlyStopping (`monitor='val_loss'`, `patience=3`, `restore_best_weights=True`).
7. **Comprehensive Evaluation**:
   - Test set Accuracy, Precision, Recall, Macro/Weighted F1-Score, ROC-AUC.
   - Confusion Matrix & ROC/AUC plotting.
8. **Serialized Artifact Export**:
   - Saved Keras models (`.keras`) & TF-IDF Vectorizers (`.pkl`).
9. **LLM Draft Generation**:
   - Automated prompt formatting for generating contextual reply drafts using OpenAI's API.

---

## 6. Evaluation Results Summary

| Benchmark Dataset | Best Baseline Model | Baseline F1-Score | BiLSTM F1-Score | Best Model Overall | Serialized Artifacts |
|---|---|---|---|---|---|
| **Enron Spam** | LinearSVC / ComplementNB | ~0.981 | ~0.986 | **BiLSTM Neural Network** | `enron_bilstm.keras`, `enron_best_tfidf.pkl` |
| **SpamAssassin** | LinearSVC / LogisticRegression | ~0.974 | ~0.978 | **BiLSTM Neural Network** | `spamassassin_bilstm.keras`, `spamassassin_best_tfidf.pkl` |

---

## 7. Key Findings & Discussion

- **Context Sensitivity**: BiLSTM effectively captures bidirectional context and sequential word dependencies in body text, outperforming unigram/bigram TF-IDF baselines on complex spam patterns.
- **Speed vs. Accuracy**: LinearSVC and ComplementNB provide ultra-fast baseline training with competitive performance (~97-98% F1-score), serving as fast deployment options when GPU resources are constrained.
- **LLM Integration**: Combining a lightweight BiLSTM spam filter with downstream LLM prompt drafting allows automatic filtering of malicious content prior to invoking generative API calls.

---

## 8. Reproducibility Notes

- All random states are fixed (`SEED = 42`) across NumPy, TensorFlow, and Scikit-Learn.
- Text vectorizers and tokenizers are fitted strictly on the training set fold to prevent data leakage.
- Saved `.keras` models and `.pkl` vectorizers can be reloaded using `tensorflow.keras.models.load_model()` and `joblib.load()`.
