# -*- coding: utf-8 -*-
"""
Converted from Dataset1_Tweet_Sentiment.ipynb
"""

# Helper for environments where display() is not built-in
try:
    display
except NameError:
    display = print


# %% [markdown]
# # Tweet Sentiment Analysis — Dataset 1
# ### Source: `cardiffnlp/tweet_eval` — Sentiment Subset
# **Labels:** negative (0) · neutral (1) · positive (2)  
# **Splits:** Official train / validation / test provided by HuggingFace


# %% [markdown]
# ## 1. Install & Import Libraries


# %% [code] (Cell 2)
# !pip install -q datasets vaderSentiment  # Magic command commented out


# %% [code] (Cell 3)
import os, sys, re, time, random, warnings, pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import joblib
from collections import Counter

from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix)

import nltk
nltk.download('vader_lexicon', quiet=True)
nltk.download('stopwords', quiet=True)
from nltk.sentiment.vader import SentimentIntensityAnalyzer

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

from datasets import load_dataset

warnings.filterwarnings('ignore')
matplotlib.rcParams['figure.dpi'] = 120
print('Libraries loaded.')


# %% [markdown]
# ## 2. Configuration & Reproducibility


# %% [code] (Cell 5)
RANDOM_SEED   = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
tf.random.set_seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)

# Column names — change if needed
TEXT_COLUMN   = 'text'
LABEL_COLUMN  = 'label'
ENTITY_COLUMN = None   # No entity column in this dataset

# BiLSTM hyperparameters
VOCAB_SIZE    = 20000
EMBEDDING_DIM = 128
LSTM_UNITS    = 64
DROPOUT_RATE  = 0.3
BATCH_SIZE    = 64
MAX_EPOCHS    = 15
LEARNING_RATE = 1e-3

gpus = tf.config.list_physical_devices('GPU')
DEVICE = 'GPU' if gpus else 'CPU'
print(f'Device: {DEVICE}')
print(f'TensorFlow: {tf.__version__}')
print(f'Python: {sys.version}')


# %% [markdown]
# ## 3. Create Output Directories


# %% [code] (Cell 7)
OUTPUT_DIR = 'outputs_dataset1'
for sub in ['figures', 'CSV', 'models', 'metrics']:
    os.makedirs(os.path.join(OUTPUT_DIR, sub), exist_ok=True)
print(f'Output directory ready: {OUTPUT_DIR}/')


# %% [markdown]
# ## 4. Load Dataset


# %% [code] (Cell 9)
print('Loading cardiffnlp/tweet_eval — sentiment subset ...')
dataset = load_dataset('cardiffnlp/tweet_eval', 'sentiment')
print('Available splits:', list(dataset.keys()))
print(dataset)


# %% [code] (Cell 10)
train_df = dataset['train'].to_pandas()
val_df   = dataset['validation'].to_pandas()
test_df  = dataset['test'].to_pandas()
print(f'Train:      {train_df.shape}')
print(f'Validation: {val_df.shape}')
print(f'Test:       {test_df.shape}')


# %% [markdown]
# ## 5. Dataset Overview


# %% [code] (Cell 12)
label_names = dataset['train'].features['label'].names
print('Label mapping:')
for i, name in enumerate(label_names):
    print(f'  {i} -> {name}')
NUM_CLASSES = len(label_names)
print(f'Number of classes: {NUM_CLASSES}')
print(f'Class names: {label_names}')


# %% [code] (Cell 13)
for df in [train_df, val_df, test_df]:
    df['sentiment'] = df[LABEL_COLUMN].map(lambda x: label_names[x])

print('Columns:', train_df.columns.tolist())
print('Dtypes:')
print(train_df.dtypes)
print('\nFirst 5 rows:')
display(train_df.head())


# %% [code] (Cell 14)
print('Random sample (train):')
display(train_df.sample(5, random_state=RANDOM_SEED))


# %% [code] (Cell 15)
print('Missing values (train):')
print(train_df.isnull().sum())
print(f'\nDuplicate rows: {train_df.duplicated().sum()}')
print(f'Duplicate texts: {train_df.duplicated(subset=[TEXT_COLUMN]).sum()}')


# %% [code] (Cell 16)
dist = train_df['sentiment'].value_counts().reset_index()
dist.columns = ['sentiment', 'count']
print('Training label distribution:')
print(dist.to_string(index=False))

total_samples = len(train_df) + len(val_df) + len(test_df)
print(f'\nTotal samples: {total_samples}')
print(f'  Train:      {len(train_df)}')
print(f'  Validation: {len(val_df)}')
print(f'  Test:       {len(test_df)}')

dist.to_csv(f'{OUTPUT_DIR}/CSV/class_distribution.csv', index=False)
summary = pd.DataFrame({'Split': ['train','validation','test','total'],
                        'Samples': [len(train_df),len(val_df),len(test_df),total_samples]})
summary.to_csv(f'{OUTPUT_DIR}/CSV/dataset_summary.csv', index=False)
print('Saved dataset_summary.csv and class_distribution.csv')


# %% [markdown]
# ## 6. Data Cleaning & Preprocessing


# %% [code] (Cell 18)
def clean_tweet(text):
    """Minimal preprocessing — preserves emojis, hashtags, negation."""
    if not isinstance(text, str):
        return ''
    text = re.sub(r'http\S+|www\.\S+', '<URL>', text)
    text = re.sub(r'@\w+', '<USER>', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

for df in [train_df, val_df, test_df]:
    df['clean_text'] = df[TEXT_COLUMN].apply(clean_tweet)

for df in [train_df, val_df, test_df]:
    bad = df['clean_text'].str.len() == 0
    df.drop(index=df[bad].index, inplace=True)
    df.reset_index(drop=True, inplace=True)

print(f'After cleaning -> train: {len(train_df)}, val: {len(val_df)}, test: {len(test_df)}')
print('\nSample cleaned tweets:')
for _, row in train_df.sample(3, random_state=RANDOM_SEED).iterrows():
    print(f'  [{row["sentiment"]}] {row["clean_text"]}')


# %% [markdown]
# ## 7. Exploratory Data Analysis


# %% [code] (Cell 20)
palette = {'negative': '#e74c3c', 'neutral': '#95a5a6', 'positive': '#27ae60'}

# A. Sentiment distribution
fig, ax = plt.subplots(figsize=(7, 4))
dist_s = train_df['sentiment'].value_counts()
bars = ax.bar(dist_s.index, dist_s.values,
              color=[palette.get(l, '#3498db') for l in dist_s.index], edgecolor='white')
for bar, val in zip(bars, dist_s.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, str(val),
            ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Sentiment Distribution (Training Set)', fontsize=13, fontweight='bold')
ax.set_xlabel('Sentiment'); ax.set_ylabel('Count')
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/class_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: class_distribution.png')


# %% [code] (Cell 21)
# B & C. Tweet length
for df in [train_df, val_df, test_df]:
    df['tweet_length'] = df['clean_text'].str.split().str.len()

fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(train_df['tweet_length'], bins=40, color='#3498db', edgecolor='white', alpha=0.85)
axes[0].set_title('Tweet Length Distribution', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Number of Words'); axes[0].set_ylabel('Frequency')
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

for label, grp in train_df.groupby('sentiment'):
    axes[1].hist(grp['tweet_length'], bins=30, alpha=0.6,
                 label=label, color=palette.get(label, '#3498db'), edgecolor='none')
axes[1].legend()
axes[1].set_title('Tweet Length by Sentiment', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Number of Words'); axes[1].set_ylabel('Frequency')
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/tweet_length_distribution.png', dpi=150, bbox_inches='tight')
plt.show()

print('Length stats by sentiment:')
print(train_df.groupby('sentiment')['tweet_length'].describe().round(1).to_string())


# %% [code] (Cell 22)
# D. Top unigrams per class (training data only)
STOPWORDS = set(nltk.corpus.stopwords.words('english'))
STOPWORDS.update(['<url>', '<user>', 'url', 'user'])

def top_ngrams(texts, n=1, top_k=15):
    vec = CountVectorizer(ngram_range=(n, n), stop_words=list(STOPWORDS),
                          max_features=top_k, token_pattern=r'(?u)\b\w+\b')
    X = vec.fit_transform(texts)
    counts = X.sum(axis=0).A1
    words  = vec.get_feature_names_out()
    return sorted(zip(words, counts), key=lambda x: -x[1])[:top_k]

fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, label in zip(axes, label_names):
    subset = train_df[train_df['sentiment'] == label]['clean_text']
    ng = top_ngrams(subset, n=1, top_k=15)
    words, counts = zip(*ng)
    ax.barh(list(words)[::-1], list(counts)[::-1],
            color=palette.get(label, '#3498db'), edgecolor='white')
    ax.set_title(f'Top Words — {label}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/top_words.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: top_words.png')


# %% [code] (Cell 23)
# Top bigrams per class
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
for ax, label in zip(axes, label_names):
    subset = train_df[train_df['sentiment'] == label]['clean_text']
    ng = top_ngrams(subset, n=2, top_k=12)
    words, counts = zip(*ng)
    ax.barh(list(words)[::-1], list(counts)[::-1],
            color=palette.get(label, '#3498db'), alpha=0.85, edgecolor='white')
    ax.set_title(f'Top Bigrams — {label}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Frequency')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/top_bigrams.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: top_bigrams.png')


# %% [markdown]
# ## 8. Train / Validation / Test Split
# Using the **official** HuggingFace splits — no manual splitting needed.


# %% [code] (Cell 25)
X_train = train_df['clean_text'].values
y_train = train_df[LABEL_COLUMN].values
X_val   = val_df['clean_text'].values
y_val   = val_df[LABEL_COLUMN].values
X_test  = test_df['clean_text'].values
y_test  = test_df[LABEL_COLUMN].values

print(f'Train:      {len(X_train)} samples')
print(f'Validation: {len(X_val)} samples')
print(f'Test:       {len(X_test)} samples')
print('All models will use these same splits.')


# %% [markdown]
# ## 9. Evaluation Helper


# %% [code] (Cell 27)
all_results = {}   # accumulates all model metrics

def try_float(x):
    try: return float(x)
    except: return -1.0

def evaluate_model(name, y_true, y_pred, train_time=None, inf_time=None):
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec  = recall_score(y_true, y_pred, average='macro', zero_division=0)
    mf1  = f1_score(y_true, y_pred, average='macro', zero_division=0)
    wf1  = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    all_results[name] = {
        'Accuracy': acc, 'Macro Precision': prec, 'Macro Recall': rec,
        'Macro F1': mf1, 'Weighted F1': wf1,
        'Training Time': train_time if train_time else 'N/A',
        'Inference Time': inf_time if inf_time else 'N/A'
    }
    print(f'\n── {name} ──')
    print(f'  Accuracy:        {acc:.4f}')
    print(f'  Macro Precision: {prec:.4f}')
    print(f'  Macro Recall:    {rec:.4f}')
    print(f'  Macro F1:        {mf1:.4f}')
    print(f'  Weighted F1:     {wf1:.4f}')
    if train_time: print(f'  Training Time:   {train_time:.3f}s')
    if inf_time:   print(f'  Inference Time:  {inf_time:.4f}s')
    return acc, prec, rec, mf1, wf1


# %% [markdown]
# ## 10. Baseline — DummyClassifier


# %% [code] (Cell 29)
t0 = time.time()
dummy = DummyClassifier(strategy='most_frequent', random_state=RANDOM_SEED)
dummy.fit(X_train, y_train)
train_t = time.time() - t0
t1 = time.time()
y_dummy = dummy.predict(X_test)
inf_t = time.time() - t1
evaluate_model('DummyClassifier', y_test, y_dummy, train_t, inf_t)


# %% [markdown]
# ## 11. Baseline — VADER
# VADER compound score: > 0.05 → positive (2), < −0.05 → negative (0), else → neutral (1).  
# This aligns directly with the dataset's label structure.


# %% [code] (Cell 31)
vader = SentimentIntensityAnalyzer()

def vader_predict(texts):
    preds = []
    for text in texts:
        s = vader.polarity_scores(str(text))['compound']
        if s > 0.05:
            preds.append(2)   # positive
        elif s < -0.05:
            preds.append(0)   # negative
        else:
            preds.append(1)   # neutral
    return np.array(preds)

t0 = time.time()
y_vader = vader_predict(X_test)
inf_t = time.time() - t0
evaluate_model('VADER', y_test, y_vader, inf_time=inf_t)
print('\nClassification Report:')
print(classification_report(y_test, y_vader, target_names=label_names))


# %% [markdown]
# ## 12. TF-IDF + MultinomialNB


# %% [code] (Cell 33)
t0 = time.time()
nb_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2), max_features=50000,
        sublinear_tf=True, min_df=2
    )),
    ('classifier', MultinomialNB(alpha=0.1))
])
nb_pipe.fit(X_train, y_train)
train_t = time.time() - t0
t1 = time.time()
y_nb = nb_pipe.predict(X_test)
inf_t = time.time() - t1
evaluate_model('MultinomialNB', y_test, y_nb, train_t, inf_t)
print('\nClassification Report:')
print(classification_report(y_test, y_nb, target_names=label_names))


# %% [markdown]
# ## 13. TF-IDF + Logistic Regression


# %% [code] (Cell 35)
t0 = time.time()
lr_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2), max_features=50000,
        sublinear_tf=True, min_df=2
    )),
    ('classifier', LogisticRegression(
        C=1.0, max_iter=1000, solver='lbfgs',
        multi_class='auto', random_state=RANDOM_SEED
    ))
])
lr_pipe.fit(X_train, y_train)
train_t = time.time() - t0
t1 = time.time()
y_lr = lr_pipe.predict(X_test)
inf_t = time.time() - t1
evaluate_model('Logistic Regression', y_test, y_lr, train_t, inf_t)
print('\nClassification Report:')
print(classification_report(y_test, y_lr, target_names=label_names))


# %% [markdown]
# ## 14. TF-IDF + LinearSVC


# %% [code] (Cell 37)
t0 = time.time()
svc_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(
        ngram_range=(1, 2), max_features=50000,
        sublinear_tf=True, min_df=2
    )),
    ('classifier', LinearSVC(
        C=0.5, max_iter=2000, random_state=RANDOM_SEED
    ))
])
svc_pipe.fit(X_train, y_train)
train_t = time.time() - t0
t1 = time.time()
y_svc = svc_pipe.predict(X_test)
inf_t = time.time() - t1
evaluate_model('LinearSVC', y_test, y_svc, train_t, inf_t)
print('\nClassification Report:')
print(classification_report(y_test, y_svc, target_names=label_names))


# %% [markdown]
# ## 15. Classical Model Comparison


# %% [code] (Cell 39)
classical_df = pd.DataFrame(all_results).T.reset_index()
classical_df.columns = ['Model','Accuracy','Macro Precision','Macro Recall',
                         'Macro F1','Weighted F1','Training Time','Inference Time']
classical_df['_sort'] = classical_df['Macro F1'].apply(try_float)
classical_df = classical_df.sort_values('_sort', ascending=False).drop(columns='_sort').reset_index(drop=True)
display(classical_df.round(4))
classical_df.to_csv(f'{OUTPUT_DIR}/CSV/classical_model_comparison.csv', index=False)
print('Saved: classical_model_comparison.csv')


# %% [code] (Cell 40)
fig, ax = plt.subplots(figsize=(9, 5))
mc = classical_df['Model'].tolist()
fc = classical_df['Macro F1'].apply(try_float).tolist()
cmap_c = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(mc)))
bars = ax.bar(mc, fc, color=cmap_c, edgecolor='white')
for bar, val in zip(bars, fc):
    if val > 0:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('Classical Models — Macro F1', fontsize=13, fontweight='bold')
ax.set_ylabel('Macro F1'); ax.set_ylim(0, min(1.15, max(fc)+0.18))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/classical_model_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: classical_model_comparison.png')


# %% [markdown]
# ## 16. Model Selection
# Selecting best classical model by **validation** Macro F1.


# %% [code] (Cell 42)
classical_pipes = {
    'MultinomialNB': nb_pipe,
    'Logistic Regression': lr_pipe,
    'LinearSVC': svc_pipe
}
val_f1s = {}
for name, pipe in classical_pipes.items():
    yp = pipe.predict(X_val)
    vf1 = f1_score(y_val, yp, average='macro', zero_division=0)
    val_f1s[name] = vf1
    print(f'  {name}: Val Macro F1 = {vf1:.4f}')

best_classical_name = max(val_f1s, key=val_f1s.get)
best_classical_pipe = classical_pipes[best_classical_name]
print(f'\nSelected classical model: {best_classical_name}')
print(f'Validation Macro F1: {val_f1s[best_classical_name]:.4f}')

joblib.dump(best_classical_pipe, f'{OUTPUT_DIR}/models/selected_classical_pipeline.joblib')
print('Saved: selected_classical_pipeline.joblib')


# %% [code] (Cell 43)
y_best_classical = best_classical_pipe.predict(X_test)
print(f'Test Classification Report — {best_classical_name}')
print(classification_report(y_test, y_best_classical, target_names=label_names))

cr_df = pd.DataFrame(
    classification_report(y_test, y_best_classical, target_names=label_names, output_dict=True)
).T.reset_index()
cr_df.to_csv(f'{OUTPUT_DIR}/CSV/classification_report.csv', index=False)

preds_df = pd.DataFrame({
    'text': X_test,
    'actual_label': [label_names[i] for i in y_test],
    'predicted_label': [label_names[i] for i in y_best_classical]
})
preds_df.to_csv(f'{OUTPUT_DIR}/CSV/test_predictions.csv', index=False)
print('Saved: classification_report.csv, test_predictions.csv')


# %% [code] (Cell 44)
cm = confusion_matrix(y_test, y_best_classical)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=label_names, yticklabels=label_names)
axes[0].set_title(f'{best_classical_name} — Confusion Matrix', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
cm_n = cm.astype(float) / cm.sum(axis=1, keepdims=True)
sns.heatmap(cm_n, annot=True, fmt='.2f', cmap='Blues', ax=axes[1],
            xticklabels=label_names, yticklabels=label_names)
axes[1].set_title(f'{best_classical_name} — Normalized', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/figures/normalized_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: confusion_matrix.png')


# %% [markdown]
# ## 17. Preprocessing Ablation


# %% [code] (Cell 46)
def clean_aggressive(text):
    """More normalized: lowercase, strip punctuation."""
    if not isinstance(text, str): return ''
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s#!?]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

X_train_agg = [clean_aggressive(t) for t in train_df[TEXT_COLUMN].values]
X_test_agg  = [clean_aggressive(t) for t in test_df[TEXT_COLUMN].values]

pipe_min = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000,
                                                sublinear_tf=True, min_df=2)),
                     ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED))])
pipe_agg = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000,
                                                sublinear_tf=True, min_df=2)),
                     ('clf', LogisticRegression(C=1.0, max_iter=1000, random_state=RANDOM_SEED))])
pipe_min.fit(X_train, y_train)
pipe_agg.fit(X_train_agg, y_train)

f1_min = f1_score(y_test, pipe_min.predict(X_test),     average='macro', zero_division=0)
f1_agg = f1_score(y_test, pipe_agg.predict(X_test_agg), average='macro', zero_division=0)

ablation_df = pd.DataFrame({
    'Preprocessing': ['Minimal (URL+mention replace)', 'Normalized (lowercase+strip)'],
    'Macro F1': [f1_min, f1_agg]
})
print(ablation_df.to_string(index=False))
ablation_df.to_csv(f'{OUTPUT_DIR}/CSV/preprocessing_comparison.csv', index=False)
print('Saved: preprocessing_comparison.csv')


# %% [code] (Cell 47)
fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(ablation_df['Preprocessing'], ablation_df['Macro F1'],
              color=['#3498db', '#e67e22'], edgecolor='white')
for bar, val in zip(bars, ablation_df['Macro F1']):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003,
            f'{val:.4f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
ax.set_title('Preprocessing Comparison (Macro F1)', fontsize=12, fontweight='bold')
ax.set_ylabel('Macro F1')
ax.set_ylim(0, min(1.1, max(ablation_df['Macro F1'])+0.15))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/preprocessing_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: preprocessing_comparison.png')


# %% [markdown]
# ## 18. BiLSTM — Data Preparation


# %% [code] (Cell 49)
tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

train_lens = [len(t.split()) for t in X_train]
MAX_SEQ_LEN = int(np.percentile(train_lens, 95))
print(f'Vocabulary size: {VOCAB_SIZE}')
print(f'Max sequence length (95th percentile): {MAX_SEQ_LEN}')

def encode_texts(texts, tok, max_len):
    seqs = tok.texts_to_sequences(texts)
    return pad_sequences(seqs, maxlen=max_len, padding='post', truncating='post')

X_train_seq = encode_texts(X_train, tokenizer, MAX_SEQ_LEN)
X_val_seq   = encode_texts(X_val,   tokenizer, MAX_SEQ_LEN)
X_test_seq  = encode_texts(X_test,  tokenizer, MAX_SEQ_LEN)

print(f'X_train_seq: {X_train_seq.shape}')
print(f'X_val_seq:   {X_val_seq.shape}')
print(f'X_test_seq:  {X_test_seq.shape}')

with open(f'{OUTPUT_DIR}/models/bilstm_tokenizer.pkl', 'wb') as f:
    pickle.dump(tokenizer, f)
print('Saved: bilstm_tokenizer.pkl')


# %% [markdown]
# ## 19. BiLSTM — Architecture


# %% [code] (Cell 51)
def build_bilstm(vocab_size, emb_dim, lstm_units, dropout, seq_len, n_classes):
    model = Sequential([
        Input(shape=(seq_len,)),
        Embedding(input_dim=vocab_size, output_dim=emb_dim, mask_zero=True),
        Bidirectional(LSTM(lstm_units, return_sequences=False)),
        Dropout(dropout),
        Dense(64, activation='relu'),
        Dropout(dropout / 2),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

bilstm = build_bilstm(VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS, DROPOUT_RATE, MAX_SEQ_LEN, NUM_CLASSES)
bilstm.summary()


# %% [markdown]
# ## 20. BiLSTM — Training


# %% [code] (Cell 53)
CKPT_PATH = f'{OUTPUT_DIR}/models/bilstm_best.weights.h5'
callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=1),
    ModelCheckpoint(CKPT_PATH, monitor='val_accuracy',
                    save_best_only=True, save_weights_only=True, verbose=0)
]

print(f'Training BiLSTM on {DEVICE} ...')
t0 = time.time()
history = bilstm.fit(
    X_train_seq, y_train,
    validation_data=(X_val_seq, y_val),
    epochs=MAX_EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)
bilstm_train_time = time.time() - t0
best_epoch = int(np.argmax(history.history['val_accuracy'])) + 1
print(f'\nTraining time: {bilstm_train_time:.1f}s  |  Best epoch: {best_epoch}')


# %% [markdown]
# ## 21. BiLSTM — Learning Curves


# %% [code] (Cell 55)
epochs_ran = range(1, len(history.history['loss']) + 1)
fig, axes = plt.subplots(1, 2, figsize=(13, 4))

axes[0].plot(epochs_ran, history.history['loss'],     'b-o', ms=4, label='Train Loss')
axes[0].plot(epochs_ran, history.history['val_loss'], 'r-o', ms=4, label='Val Loss')
axes[0].axvline(best_epoch, ls='--', color='gray', alpha=0.7, label=f'Best ({best_epoch})')
axes[0].set_title('Loss Curves', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss'); axes[0].legend()
axes[0].spines['top'].set_visible(False); axes[0].spines['right'].set_visible(False)

axes[1].plot(epochs_ran, history.history['accuracy'],     'b-o', ms=4, label='Train Accuracy')
axes[1].plot(epochs_ran, history.history['val_accuracy'], 'r-o', ms=4, label='Val Accuracy')
axes[1].axvline(best_epoch, ls='--', color='gray', alpha=0.7, label=f'Best ({best_epoch})')
axes[1].set_title('Accuracy Curves', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy'); axes[1].legend()
axes[1].spines['top'].set_visible(False); axes[1].spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/bilstm_loss_curve.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/figures/bilstm_accuracy_curve.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: bilstm_loss_curve.png, bilstm_accuracy_curve.png')


# %% [markdown]
# ## 22. BiLSTM — Test Evaluation


# %% [code] (Cell 57)
t0 = time.time()
y_bilstm_proba = bilstm.predict(X_test_seq, batch_size=BATCH_SIZE, verbose=0)
bilstm_inf_time = time.time() - t0
y_bilstm     = np.argmax(y_bilstm_proba, axis=1)
bilstm_conf  = np.max(y_bilstm_proba, axis=1)

evaluate_model('BiLSTM', y_test, y_bilstm, train_time=bilstm_train_time, inf_time=bilstm_inf_time)
print('\nClassification Report:')
print(classification_report(y_test, y_bilstm, target_names=label_names))

bilstm_m = {
    'Accuracy': accuracy_score(y_test, y_bilstm),
    'Macro Precision': precision_score(y_test, y_bilstm, average='macro', zero_division=0),
    'Macro Recall':    recall_score(y_test, y_bilstm,    average='macro', zero_division=0),
    'Macro F1':        f1_score(y_test, y_bilstm,        average='macro', zero_division=0),
    'Weighted F1':     f1_score(y_test, y_bilstm,        average='weighted', zero_division=0),
    'Training Time': bilstm_train_time,
    'Best Epoch': best_epoch
}
pd.DataFrame([bilstm_m]).to_csv(f'{OUTPUT_DIR}/CSV/bilstm_metrics.csv', index=False)

bilstm_cr = pd.DataFrame(
    classification_report(y_test, y_bilstm, target_names=label_names, output_dict=True)
).T.reset_index()
bilstm_cr.to_csv(f'{OUTPUT_DIR}/CSV/bilstm_classification_report.csv', index=False)
print('Saved: bilstm_metrics.csv, bilstm_classification_report.csv')


# %% [code] (Cell 58)
cm_b = confusion_matrix(y_test, y_bilstm)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cm_b, annot=True, fmt='d', cmap='Purples', ax=axes[0],
            xticklabels=label_names, yticklabels=label_names)
axes[0].set_title('BiLSTM — Confusion Matrix', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
cm_bn = cm_b.astype(float) / cm_b.sum(axis=1, keepdims=True)
sns.heatmap(cm_bn, annot=True, fmt='.2f', cmap='Purples', ax=axes[1],
            xticklabels=label_names, yticklabels=label_names)
axes[1].set_title('BiLSTM — Normalized Confusion Matrix', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/bilstm_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.savefig(f'{OUTPUT_DIR}/figures/bilstm_normalized_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: bilstm_confusion_matrix.png')


# %% [code] (Cell 59)
pd.DataFrame({
    'text': X_test,
    'actual_label': [label_names[i] for i in y_test],
    'predicted_label': [label_names[i] for i in y_bilstm],
    'confidence': bilstm_conf
}).to_csv(f'{OUTPUT_DIR}/CSV/bilstm_test_predictions.csv', index=False)
print('Saved: bilstm_test_predictions.csv')


# %% [markdown]
# ## 23. BiLSTM — Error Analysis


# %% [code] (Cell 61)
err_mask = y_test != y_bilstm
err_df = pd.DataFrame({
    'text': X_test[err_mask],
    'actual_label':    [label_names[i] for i in y_test[err_mask]],
    'predicted_label': [label_names[i] for i in y_bilstm[err_mask]],
    'confidence': bilstm_conf[err_mask]
}).sort_values('confidence', ascending=False).reset_index(drop=True)

print(f'Misclassifications: {len(err_df)} / {len(y_test)}  ({len(err_df)/len(y_test)*100:.1f}%)')
err_df.to_csv(f'{OUTPUT_DIR}/CSV/bilstm_error_analysis.csv', index=False)
print('Saved: bilstm_error_analysis.csv')
print('\nSample errors (high confidence):')
display(err_df.head(10))


# %% [code] (Cell 62)
print('Error breakdown (actual → predicted):')
print(err_df.groupby(['actual_label','predicted_label']).size()
      .reset_index(name='count').to_string(index=False))


# %% [markdown]
# ## 24. Classical vs BiLSTM — Full Comparison


# %% [code] (Cell 64)
final_df = pd.DataFrame(all_results).T.reset_index()
final_df.columns = ['Model','Accuracy','Macro Precision','Macro Recall',
                     'Macro F1','Weighted F1','Training Time','Inference Time']
final_df['_sort'] = final_df['Macro F1'].apply(try_float)
final_df = final_df.sort_values('_sort', ascending=False).drop(columns='_sort').reset_index(drop=True)
display(final_df.round(4))
final_df.to_csv(f'{OUTPUT_DIR}/CSV/final_model_comparison.csv', index=False)
print('Saved: final_model_comparison.csv')


# %% [code] (Cell 65)
fig, ax = plt.subplots(figsize=(10, 5))
ml = final_df['Model'].tolist()
fl = final_df['Macro F1'].apply(try_float).tolist()
cl = ['#8e44ad' if m == 'BiLSTM' else '#3498db' for m in ml]
bars = ax.bar(ml, fl, color=cl, edgecolor='white')
for bar, val in zip(bars, fl):
    if val > 0:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
ax.set_title('All Models — Macro F1 Comparison', fontsize=13, fontweight='bold')
ax.set_ylabel('Macro F1'); ax.set_ylim(0, min(1.15, max(fl)+0.18))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.xticks(rotation=15, ha='right')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/final_macro_f1_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: final_macro_f1_comparison.png')


# %% [markdown]
# ## 25. Entity / Product Analysis


# %% [code] (Cell 67)
if ENTITY_COLUMN is None:
    print('Entity-level analysis is not applicable to this dataset.')
    print('The cardiffnlp/tweet_eval sentiment subset does not include a product, brand, or entity column.')


# %% [markdown]
# ## 26. Final Error Analysis


# %% [code] (Cell 69)
best_model_name = final_df.iloc[0]['Model']
best_model_f1   = try_float(final_df.iloc[0]['Macro F1'])

pred_map = {
    'BiLSTM': y_bilstm,
    'Logistic Regression': y_lr,
    'LinearSVC': y_svc,
    'MultinomialNB': y_nb,
    'DummyClassifier': y_dummy,
    'VADER': y_vader
}
y_best = pred_map.get(best_model_name, y_lr)

errors_idx = np.where(y_test != y_best)[0]
error_final = pd.DataFrame({
    'text': X_test[errors_idx],
    'actual_label':    [label_names[i] for i in y_test[errors_idx]],
    'predicted_label': [label_names[i] for i in y_best[errors_idx]]
}).reset_index(drop=True)

print(f'Best model: {best_model_name}  (Macro F1: {best_model_f1:.4f})')
print(f'Misclassifications on test set: {len(error_final)} / {len(y_test)}')
error_final.to_csv(f'{OUTPUT_DIR}/CSV/error_analysis.csv', index=False)
print('Saved: error_analysis.csv')
print('\nSample misclassified tweets:')
display(error_final.sample(min(10, len(error_final)), random_state=RANDOM_SEED))


# %% [markdown]
# ## 27. Save BiLSTM Model


# %% [code] (Cell 71)
bilstm.save(f'{OUTPUT_DIR}/models/bilstm_model.keras')
print('Saved: bilstm_model.keras')


# %% [markdown]
# ## 28. Run Information


# %% [code] (Cell 73)
import platform

run_info = (
    f'Dataset: cardiffnlp/tweet_eval - sentiment\n'
    f'Number of samples: {total_samples}\n'
    f'Number of classes: {NUM_CLASSES}\n'
    f'Labels: {label_names}\n'
    f'Random seed: {RANDOM_SEED}\n'
    f'Device: {DEVICE}\n'
    f'Train samples: {len(X_train)}\n'
    f'Validation samples: {len(X_val)}\n'
    f'Test samples: {len(X_test)}\n'
    f'Python version: {sys.version}\n'
    f'TensorFlow: {tf.__version__}\n'
    f'\nBiLSTM Config:\n'
    f'  Vocab size: {VOCAB_SIZE}\n'
    f'  Max seq length: {MAX_SEQ_LEN}\n'
    f'  Embedding dim: {EMBEDDING_DIM}\n'
    f'  LSTM units: {LSTM_UNITS}\n'
    f'  Dropout: {DROPOUT_RATE}\n'
    f'  Batch size: {BATCH_SIZE}\n'
    f'  Best epoch: {best_epoch}\n'
    f'  Training time: {bilstm_train_time:.1f}s\n'
    f'\nBest model: {best_model_name}\n'
    f'Best Macro F1: {best_model_f1:.4f}\n'
)
with open(f'{OUTPUT_DIR}/metrics/run_information.txt', 'w') as f:
    f.write(run_info)
print(run_info)


# %% [markdown]
# ## 29. Final Results


# %% [code] (Cell 75)
disp = final_df[['Model','Accuracy','Macro F1','Weighted F1']].copy()
for col in ['Accuracy','Macro F1','Weighted F1']:
    disp[col] = disp[col].apply(lambda x: f'{try_float(x):.4f}' if try_float(x) >= 0 else str(x))

print('=' * 60)
print('FINAL RESULTS — Dataset 1')
print('=' * 60)
print(disp.to_string(index=False))
print('=' * 60)
print(f'Best Model:    {best_model_name}')
print(f'Best Macro F1: {best_model_f1:.4f}')
print(f'Best Accuracy: {try_float(final_df.iloc[0]["Accuracy"]):.4f}')


# %% [markdown]
# ## 30. Final Observations


# %% [code] (Cell 77)
cl_names = ['MultinomialNB', 'Logistic Regression', 'LinearSVC']
cl_f1s   = {m: try_float(all_results[m]['Macro F1']) for m in cl_names if m in all_results}
best_cl  = max(cl_f1s, key=cl_f1s.get)
bilstm_f1_val = try_float(all_results.get('BiLSTM', {}).get('Macro F1', -1))

cr_dict  = classification_report(y_test, y_best, target_names=label_names, output_dict=True)
cls_f1s  = {c: cr_dict[c]['f1-score'] for c in label_names}
hardest  = min(cls_f1s, key=cls_f1s.get)

print('Final Observations')
print('-' * 55)
print(f'Best overall:    {best_model_name}  (Macro F1 = {best_model_f1:.4f})')
print(f'Best classical:  {best_cl}  (Macro F1 = {cl_f1s[best_cl]:.4f})')
if bilstm_f1_val >= 0:
    diff = bilstm_f1_val - cl_f1s[best_cl]
    word = 'improved over' if diff > 0 else 'did not surpass'
    print(f'BiLSTM {word} the best classical model by {abs(diff):.4f} Macro F1.')
print(f'Hardest class to predict: "{hardest}"  (F1 = {cls_f1s[hardest]:.4f})')
print(f'Total test errors:  {len(error_final)} / {len(y_test)}')
print('-' * 55)


# %% [markdown]
# ---
# ## Advanced Experiment A — Stratified 5-Fold Cross-Validation
# Using the same folds across MultinomialNB, Logistic Regression, and LinearSVC. Validation and test sets from the original experiment are untouched.


# %% [code] (Cell 79)
from sklearn.model_selection import StratifiedKFold, cross_validate
import warnings
warnings.filterwarnings('ignore')

SKF = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

# Pool train+val for CV (test remains locked)
X_cv = np.concatenate([X_train, X_val])
y_cv = np.concatenate([y_train, y_val])

cv_pipes = {
    'MultinomialNB':      nb_pipe,
    'Logistic Regression': lr_pipe,
    'LinearSVC':          svc_pipe,
}

cv_rows = []
for name, pipe in cv_pipes.items():
    t0 = time.time()
    scores = cross_validate(
        pipe, X_cv, y_cv,
        cv=SKF,
        scoring={'macro_f1': 'f1_macro', 'weighted_f1': 'f1_weighted'},
        n_jobs=-1, return_train_score=False
    )
    elapsed = time.time() - t0
    mf1_mean = scores['test_macro_f1'].mean()
    mf1_std  = scores['test_macro_f1'].std()
    wf1_mean = scores['test_weighted_f1'].mean()
    wf1_std  = scores['test_weighted_f1'].std()
    cv_rows.append({
        'Model': name,
        'CV Macro F1 Mean': round(mf1_mean, 4),
        'CV Macro F1 SD':   round(mf1_std,  4),
        'CV Weighted F1 Mean': round(wf1_mean, 4),
        'CV Weighted F1 SD':   round(wf1_std,  4),
        'CV Training Time (s)': round(elapsed, 3),
        'Fold Scores': scores['test_macro_f1'].tolist()
    })
    print(f'{name}: Macro F1 = {mf1_mean:.4f} +/- {mf1_std:.4f}  |  Weighted F1 = {wf1_mean:.4f} +/- {wf1_std:.4f}')

cv_df = pd.DataFrame(cv_rows)
cv_df_save = cv_df.drop(columns=['Fold Scores'])
cv_df_save.to_csv(f'{OUTPUT_DIR}/CSV/cv_results_D1.csv', index=False)
print('\nSaved: cv_results_D1.csv')
display(cv_df_save)


# %% [code] (Cell 80)
fig, ax = plt.subplots(figsize=(8, 4))
means = cv_df['CV Macro F1 Mean'].values
sds   = cv_df['CV Macro F1 SD'].values
models_cv = cv_df['Model'].tolist()
ax.bar(models_cv, means, yerr=sds, capsize=6,
       color=['#3498db','#2ecc71','#e74c3c'], edgecolor='white', alpha=0.9)
for i,(m,s) in enumerate(zip(means, sds)):
    ax.text(i, m+s+0.005, f'{m:.4f}', ha='center', fontsize=9, fontweight='bold')
ax.set_title('5-Fold CV Macro F1 (mean ± SD) — Dataset 1', fontsize=12, fontweight='bold')
ax.set_ylabel('Macro F1'); ax.set_ylim(0, min(1.1, max(means+sds)+0.12))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/cv_macro_f1_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: cv_macro_f1_comparison.png')


# %% [markdown]
# ## Advanced Experiment B — Character TF-IDF
# Adding character-level TF-IDF as a new representation. Existing word TF-IDF models are **not** replaced.


# %% [code] (Cell 82)
from sklearn.pipeline import Pipeline as SKPipeline

char_pipe = SKPipeline([
    ('tfidf', TfidfVectorizer(
        analyzer='char_wb', ngram_range=(2, 5),
        max_features=80000, sublinear_tf=True, min_df=2
    )),
    ('classifier', LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_SEED))
])

# CV comparison
char_cv_scores = cross_validate(
    char_pipe, X_cv, y_cv, cv=SKF,
    scoring={'macro_f1': 'f1_macro', 'weighted_f1': 'f1_weighted'},
    n_jobs=-1
)
word_cv_scores = cross_validate(
    svc_pipe, X_cv, y_cv, cv=SKF,
    scoring={'macro_f1': 'f1_macro', 'weighted_f1': 'f1_weighted'},
    n_jobs=-1
)

# Train on full train set and evaluate on test
t0 = time.time()
char_pipe.fit(X_train, y_train)
char_train_time = time.time() - t0
t0 = time.time()
y_char = char_pipe.predict(X_test)
char_inf_time = time.time() - t0

char_acc  = accuracy_score(y_test, y_char)
char_mf1  = f1_score(y_test, y_char, average='macro',    zero_division=0)
char_wf1  = f1_score(y_test, y_char, average='weighted', zero_division=0)

# Word SVC test metrics
word_acc = accuracy_score(y_test, y_svc)
word_mf1 = f1_score(y_test, y_svc, average='macro',    zero_division=0)
word_wf1 = f1_score(y_test, y_svc, average='weighted', zero_division=0)

char_results = pd.DataFrame([
    {'Representation': 'Word TF-IDF (1-2gram)',
     'CV Macro F1': round(word_cv_scores['test_macro_f1'].mean(), 4),
     'CV SD':       round(word_cv_scores['test_macro_f1'].std(),  4),
     'Test Macro F1': round(word_mf1, 4),
     'Test Weighted F1': round(word_wf1, 4),
     'Test Accuracy': round(word_acc, 4)},
    {'Representation': 'Char TF-IDF (2-5gram)',
     'CV Macro F1': round(char_cv_scores['test_macro_f1'].mean(), 4),
     'CV SD':       round(char_cv_scores['test_macro_f1'].std(),  4),
     'Test Macro F1': round(char_mf1, 4),
     'Test Weighted F1': round(char_wf1, 4),
     'Test Accuracy': round(char_acc, 4)},
])
print('Character vs Word TF-IDF Comparison:')
display(char_results)
char_results.to_csv(f'{OUTPUT_DIR}/CSV/character_tfidf_results_D1.csv', index=False)
print('Saved: character_tfidf_results_D1.csv')
print('\nChar TF-IDF Classification Report:')
print(classification_report(y_test, y_char, target_names=label_names))


# %% [code] (Cell 83)
fig, ax = plt.subplots(figsize=(7, 4))
labels_rep = char_results['Representation'].tolist()
cv_means = char_results['CV Macro F1'].values
cv_sds   = char_results['CV SD'].values
test_mf1 = char_results['Test Macro F1'].values
x = np.arange(len(labels_rep))
w = 0.35
b1 = ax.bar(x - w/2, cv_means,  w, yerr=cv_sds,  capsize=5, label='CV Macro F1',   color='#3498db', alpha=0.85)
b2 = ax.bar(x + w/2, test_mf1,  w,               label='Test Macro F1', color='#e74c3c', alpha=0.85)
ax.set_xticks(x); ax.set_xticklabels(labels_rep, rotation=10)
ax.set_title('Word vs Character TF-IDF — Macro F1', fontsize=12, fontweight='bold')
ax.set_ylabel('Macro F1'); ax.legend()
ax.set_ylim(0, min(1.1, max(np.concatenate([cv_means, test_mf1]))+0.15))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/character_tfidf_comparison.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: character_tfidf_comparison.png')


# %% [markdown]
# ## Advanced Experiment C — BiLSTM Multi-Seed Uncertainty
# The original BiLSTM (seed 42) remains unchanged above. This section runs the same architecture with seeds 42, 123, 2026 to estimate variance.


# %% [code] (Cell 85)
import scipy.stats as stats

MULTISEED_SEEDS = [42, 123, 2026]
ms_rows = []

for seed_val in MULTISEED_SEEDS:
    print(f'\n--- BiLSTM Seed {seed_val} ---')
    tf.random.set_seed(seed_val)
    np.random.seed(seed_val)
    random.seed(seed_val)

    ms_model = build_bilstm(VOCAB_SIZE, EMBEDDING_DIM, LSTM_UNITS, DROPOUT_RATE, MAX_SEQ_LEN, NUM_CLASSES)
    ms_ckpt  = f'{OUTPUT_DIR}/models/bilstm_seed{seed_val}.weights.h5'
    ms_cb = [
        EarlyStopping(monitor='val_accuracy', patience=3, restore_best_weights=True, verbose=0),
        ModelCheckpoint(ms_ckpt, monitor='val_accuracy', save_best_only=True, save_weights_only=True, verbose=0)
    ]
    t0 = time.time()
    ms_hist = ms_model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=ms_cb, verbose=0
    )
    train_t = time.time() - t0

    best_ep  = int(np.argmax(ms_hist.history['val_accuracy'])) + 1
    val_mf1  = f1_score(y_val,  np.argmax(ms_model.predict(X_val_seq,  verbose=0), axis=1), average='macro', zero_division=0)
    t1 = time.time()
    y_ms_pred = np.argmax(ms_model.predict(X_test_seq, verbose=0), axis=1)
    inf_t = time.time() - t1
    test_mf1  = f1_score(y_test, y_ms_pred, average='macro',    zero_division=0)
    test_wf1  = f1_score(y_test, y_ms_pred, average='weighted', zero_division=0)
    test_acc  = accuracy_score(y_test, y_ms_pred)
    n_params  = ms_model.count_params()

    ms_rows.append({
        'Seed': seed_val, 'Best Epoch': best_ep,
        'Val Macro F1': round(val_mf1, 4),
        'Test Macro F1': round(test_mf1, 4),
        'Test Weighted F1': round(test_wf1, 4),
        'Test Accuracy': round(test_acc, 4),
        'Training Time (s)': round(train_t, 1),
        'Inference Time (s)': round(inf_t, 4),
        'Parameters': n_params
    })
    print(f'  Val Macro F1: {val_mf1:.4f}  |  Test Macro F1: {test_mf1:.4f}  |  Train Time: {train_t:.1f}s')
    tf.keras.backend.clear_session()

ms_df = pd.DataFrame(ms_rows)
ms_df.to_csv(f'{OUTPUT_DIR}/CSV/bilstm_multiseed_results_D1.csv', index=False)
print('\nSaved: bilstm_multiseed_results_D1.csv')

mf1_vals = ms_df['Test Macro F1'].values
mf1_mean = mf1_vals.mean()
mf1_std  = mf1_vals.std()
n = len(mf1_vals)
ci95 = stats.t.interval(0.95, df=n-1, loc=mf1_mean, scale=stats.sem(mf1_vals)) if n > 1 else (mf1_mean, mf1_mean)
print(f'\nBiLSTM Multi-Seed Summary:')
print(f'  Mean Test Macro F1: {mf1_mean:.4f}')
print(f'  Std Dev:            {mf1_std:.4f}')
print(f'  95% CI:             ({ci95[0]:.4f}, {ci95[1]:.4f})')
display(ms_df)


# %% [code] (Cell 86)
fig, ax = plt.subplots(figsize=(7, 4))
ax.bar([str(s) for s in ms_df['Seed']], ms_df['Test Macro F1'],
       color='#8e44ad', edgecolor='white', alpha=0.85)
ax.axhline(mf1_mean, color='red', linestyle='--', linewidth=1.5, label=f'Mean={mf1_mean:.4f}')
for i, val in enumerate(ms_df['Test Macro F1']):
    ax.text(i, val+0.005, f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
ax.set_title('BiLSTM Test Macro F1 by Seed', fontsize=12, fontweight='bold')
ax.set_xlabel('Seed'); ax.set_ylabel('Macro F1'); ax.legend()
ax.set_ylim(0, min(1.15, max(ms_df['Test Macro F1'])+0.15))
ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/bilstm_multiseed_results.png', dpi=150, bbox_inches='tight')
plt.show()


# %% [markdown]
# ## Advanced Experiment D — BiLSTM Model Size & Parameters
# Reporting trainable parameters, total parameters, saved model size, and inference latency for the original BiLSTM.


# %% [code] (Cell 88)
bilstm_total_params     = bilstm.count_params()
bilstm_trainable_params = sum(tf.size(v).numpy() for v in bilstm.trainable_variables)

model_keras_path = f'{OUTPUT_DIR}/models/bilstm_model.keras'
model_size_mb = os.path.getsize(model_keras_path) / (1024**2) if os.path.exists(model_keras_path) else float('nan')

# Inference latency on 100 test samples
sample_100 = X_test_seq[:100]
_ = bilstm.predict(sample_100, verbose=0)  # warmup
t0 = time.time()
for _ in range(10):
    bilstm.predict(sample_100, verbose=0)
bilstm_lat_ms = (time.time() - t0) / 10 / len(sample_100) * 1000

size_info = pd.DataFrame([{
    'Model': 'BiLSTM (seed 42)',
    'Total Parameters': bilstm_total_params,
    'Trainable Parameters': bilstm_trainable_params,
    'Model Size (MB)': round(model_size_mb, 2),
    'Inference Latency (ms/sample)': round(bilstm_lat_ms, 3),
    'Vocab Size': VOCAB_SIZE,
    'Embedding Dim': EMBEDDING_DIM,
    'LSTM Units': LSTM_UNITS,
    'Max Seq Len': MAX_SEQ_LEN,
}])
display(size_info)
size_info.to_csv(f'{OUTPUT_DIR}/CSV/bilstm_size_info_D1.csv', index=False)
print('Saved: bilstm_size_info_D1.csv')


# %% [markdown]
# ## Advanced Experiment E — Confidence Analysis
# Analysing prediction confidence (max softmax probability) for the BiLSTM and calibrated Logistic Regression. LinearSVC is excluded (no probabilities).


# %% [code] (Cell 90)
# BiLSTM confidence: already have bilstm_conf from earlier
# LR confidence: predict_proba
lr_proba = lr_pipe.predict_proba(X_test)
lr_conf  = lr_proba.max(axis=1)
lr_pred  = lr_pipe.predict(X_test)

def confidence_bin_analysis(y_true, y_pred, confidences, model_name):
    bins = [0.50, 0.60, 0.70, 0.80, 0.90, 1.001]
    bin_labels = ['0.50-0.60','0.60-0.70','0.70-0.80','0.80-0.90','0.90-1.00']
    rows = []
    for lo, hi, bl in zip(bins[:-1], bins[1:], bin_labels):
        mask = (confidences >= lo) & (confidences < hi)
        n = mask.sum()
        if n == 0:
            rows.append({'Bin': bl, 'Count': 0, 'Coverage': 0.0, 'Accuracy': float('nan'), 'Error Rate': float('nan')})
            continue
        acc = accuracy_score(y_true[mask], y_pred[mask])
        rows.append({
            'Bin': bl, 'Count': int(n),
            'Coverage': round(n/len(y_true)*100, 1),
            'Accuracy': round(acc, 4),
            'Error Rate': round(1-acc, 4)
        })
    df_bins = pd.DataFrame(rows)
    print(f'\n{model_name} — Confidence Bin Analysis:')
    display(df_bins)
    return df_bins

bilstm_bins = confidence_bin_analysis(y_test, y_bilstm, bilstm_conf, 'BiLSTM')
lr_bins     = confidence_bin_analysis(y_test, lr_pred,  lr_conf,     'Logistic Regression')


# %% [code] (Cell 91)
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Brier score & ECE for LR (has proper probabilities)
def expected_calibration_error(y_true, y_prob_max, n_bins=10):
    bins = np.linspace(0, 1, n_bins+1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (y_prob_max >= lo) & (y_prob_max < hi)
        if mask.sum() == 0: continue
        acc_bin  = accuracy_score(y_true[mask], np.array(lr_pred)[mask])
        conf_bin = y_prob_max[mask].mean()
        ece += mask.sum() / len(y_true) * abs(acc_bin - conf_bin)
    return ece

# Brier score (one-vs-rest for multiclass)
brier_scores = []
for c in range(NUM_CLASSES):
    bs = brier_score_loss((y_test == c).astype(int), lr_proba[:, c])
    brier_scores.append(bs)
brier_mean = np.mean(brier_scores)
ece_val = expected_calibration_error(y_test, lr_conf)

print(f'Logistic Regression Brier Score (mean OvR): {brier_mean:.4f}')
print(f'Logistic Regression ECE:                    {ece_val:.4f}')

# Reliability diagram for LR
fig, ax = plt.subplots(figsize=(6, 5))
for c, cname in enumerate(label_names):
    y_bin_true = (y_test == c).astype(int)
    prob_pos = lr_proba[:, c]
    try:
        frac_pos, mean_pred = calibration_curve(y_bin_true, prob_pos, n_bins=8, strategy='quantile')
        ax.plot(mean_pred, frac_pos, marker='o', label=cname)
    except Exception:
        pass
ax.plot([0,1],[0,1],'k--',alpha=0.5,label='Perfect calibration')
ax.set_title('Reliability Diagram — Logistic Regression', fontsize=11, fontweight='bold')
ax.set_xlabel('Mean Predicted Probability'); ax.set_ylabel('Fraction of Positives')
ax.legend(); ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/calibration_plot.png', dpi=150, bbox_inches='tight')
plt.show()
print('Saved: calibration_plot.png')


# %% [markdown]
# ## Advanced Experiment F — Confidence-Based Abstention
# Threshold selected **only on validation data**, then applied frozen to the test set.


# %% [code] (Cell 93)
# Choose threshold using validation data (LR probabilities)
lr_val_proba = lr_pipe.predict_proba(X_val)
lr_val_conf  = lr_val_proba.max(axis=1)
lr_val_pred  = lr_pipe.predict(X_val)

best_thresh, best_val_acc_accepted = 0.5, 0.0
for thresh in np.arange(0.50, 0.96, 0.05):
    mask = lr_val_conf >= thresh
    if mask.sum() < 10: break
    acc_acc = accuracy_score(y_val[mask], lr_val_pred[mask])
    if acc_acc > best_val_acc_accepted:
        best_val_acc_accepted = acc_acc
        best_thresh = thresh

print(f'Selected abstention threshold (from validation): {best_thresh:.2f}')
print(f'Validation accuracy on accepted (threshold {best_thresh:.2f}): {best_val_acc_accepted:.4f}')

# Apply frozen threshold to test set
accept_mask = lr_conf >= best_thresh
coverage     = accept_mask.sum() / len(y_test) * 100
auto_acc     = accuracy_score(y_test[accept_mask], lr_pred[accept_mask]) if accept_mask.sum() > 0 else float('nan')
auto_err     = 1 - auto_acc

abstention_res = pd.DataFrame([{
    'Threshold': round(best_thresh, 2),
    'Total Test Samples': len(y_test),
    'Auto-Accepted': int(accept_mask.sum()),
    'Human-Review': int((~accept_mask).sum()),
    'Coverage (%)': round(coverage, 1),
    'Auto-Accepted Accuracy': round(auto_acc, 4),
    'Auto-Accepted Error Rate': round(auto_err, 4),
    'Human-Review Rate (%)': round(100 - coverage, 1)
}])
print('\nAbstention Results (frozen test set):')
display(abstention_res)
abstention_res.to_csv(f'{OUTPUT_DIR}/CSV/abstention_results_D1.csv', index=False)
print('Saved: abstention_results_D1.csv')


# %% [markdown]
# ## Advanced Experiment G — Twitter-RoBERTa Fine-tuning
# Model: `cardiffnlp/twitter-roberta-base-sentiment-latest`  
# Uses the existing frozen train/validation/test split. The model already has 3 aligned output classes (negative=0, neutral=1, positive=2).


# %% [code] (Cell 95)
# !pip install -q transformers torch accelerate  # Magic command commented out


# %% [code] (Cell 96)
import torch
from transformers import (
    AutoTokenizer as HFTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments, Trainer,
    EarlyStoppingCallback
)
from torch.utils.data import Dataset as TorchDataset

TF_MODEL_ID = 'cardiffnlp/twitter-roberta-base-sentiment-latest'
TRANS_SEED  = 42
TRANS_LR    = 2e-5
TRANS_WD    = 0.01
TRANS_EPOCHS = 3
TRANS_MAX_LEN = 128
TRANS_BATCH  = 16

# Detect device
TRANS_DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Transformer device: {TRANS_DEVICE}')
if TRANS_DEVICE == 'cuda':
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB')


# %% [code] (Cell 97)
# twitter-roberta label map: 0=negative, 1=neutral, 2=positive — matches our label_names
TRANS_LABEL2ID = {lbl: i for i, lbl in enumerate(label_names)}
TRANS_ID2LABEL = {i: lbl for lbl, i in TRANS_LABEL2ID.items()}
print('Label mapping:', TRANS_LABEL2ID)

hf_tokenizer = HFTokenizer.from_pretrained(TF_MODEL_ID)

class SentimentDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding='max_length',
            max_length=max_len, return_tensors='pt'
        )
        self.labels = torch.tensor(labels, dtype=torch.long)
    def __len__(self): return len(self.labels)
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()} | {'labels': self.labels[idx]}

train_ds = SentimentDataset(X_train, y_train, hf_tokenizer, TRANS_MAX_LEN)
val_ds   = SentimentDataset(X_val,   y_val,   hf_tokenizer, TRANS_MAX_LEN)
test_ds  = SentimentDataset(X_test,  y_test,  hf_tokenizer, TRANS_MAX_LEN)
print(f'Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}')


# %% [code] (Cell 98)
from transformers import set_seed as hf_set_seed
hf_set_seed(TRANS_SEED)

trans_model = AutoModelForSequenceClassification.from_pretrained(
    TF_MODEL_ID,
    num_labels=NUM_CLASSES,
    id2label=TRANS_ID2LABEL,
    label2id=TRANS_LABEL2ID,
    ignore_mismatched_sizes=True
)
trans_param_count = sum(p.numel() for p in trans_model.parameters())
print(f'Twitter-RoBERTa parameters: {trans_param_count:,}')


# %% [code] (Cell 99)
import numpy as np
from sklearn.metrics import f1_score as sk_f1

def compute_metrics_hf(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=1)
    return {
        'macro_f1':    sk_f1(labels, preds, average='macro',    zero_division=0),
        'weighted_f1': sk_f1(labels, preds, average='weighted', zero_division=0),
        'accuracy':    (preds == labels).mean()
    }

TRANS_CKPT_DIR = f'{OUTPUT_DIR}/models/transformer_checkpoint_s42'

training_args = TrainingArguments(
    output_dir=TRANS_CKPT_DIR,
    num_train_epochs=TRANS_EPOCHS,
    per_device_train_batch_size=TRANS_BATCH,
    per_device_eval_batch_size=TRANS_BATCH * 2,
    learning_rate=TRANS_LR,
    weight_decay=TRANS_WD,
    evaluation_strategy='epoch',
    save_strategy='epoch',
    load_best_model_at_end=True,
    metric_for_best_model='macro_f1',
    greater_is_better=True,
    seed=TRANS_SEED,
    fp16=(TRANS_DEVICE == 'cuda'),
    logging_steps=50,
    report_to='none',
    save_total_limit=2
)

trainer = Trainer(
    model=trans_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    tokenizer=hf_tokenizer,
    compute_metrics=compute_metrics_hf,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

print(f'Fine-tuning {TF_MODEL_ID} on {TRANS_DEVICE} ...')
t0 = time.time()
trainer.train()
trans_train_time = time.time() - t0
print(f'Training complete in {trans_train_time:.1f}s')


# %% [code] (Cell 100)
# Evaluate on test set (best checkpoint)
t0 = time.time()
trans_test_preds_raw = trainer.predict(test_ds)
trans_inf_time = time.time() - t0

y_trans = np.argmax(trans_test_preds_raw.predictions, axis=1)
trans_conf = torch.softmax(torch.tensor(trans_test_preds_raw.predictions, dtype=torch.float32), dim=1).numpy().max(axis=1)

trans_acc  = accuracy_score(y_test, y_trans)
trans_prec = precision_score(y_test, y_trans, average='macro',    zero_division=0)
trans_rec  = recall_score(y_test,   y_trans,  average='macro',    zero_division=0)
trans_mf1  = f1_score(y_test,       y_trans,  average='macro',    zero_division=0)
trans_wf1  = f1_score(y_test,       y_trans,  average='weighted', zero_division=0)

print(f'Twitter-RoBERTa Test Results:')
print(f'  Accuracy:        {trans_acc:.4f}')
print(f'  Macro F1:        {trans_mf1:.4f}')
print(f'  Weighted F1:     {trans_wf1:.4f}')
print(f'  Inference Time:  {trans_inf_time:.2f}s for {len(X_test)} samples')
print()
print(classification_report(y_test, y_trans, target_names=label_names))

trans_results = pd.DataFrame([{
    'Model': 'Twitter-RoBERTa',
    'Model ID': TF_MODEL_ID,
    'Seed': TRANS_SEED,
    'Max Seq Len': TRANS_MAX_LEN,
    'Batch Size': TRANS_BATCH,
    'Learning Rate': TRANS_LR,
    'Weight Decay': TRANS_WD,
    'Epochs': TRANS_EPOCHS,
    'Parameters': trans_param_count,
    'Training Time (s)': round(trans_train_time, 1),
    'Inference Time (s)': round(trans_inf_time, 2),
    'Accuracy': round(trans_acc, 4),
    'Macro Precision': round(trans_prec, 4),
    'Macro Recall': round(trans_rec, 4),
    'Macro F1': round(trans_mf1, 4),
    'Weighted F1': round(trans_wf1, 4)
}])
trans_results.to_csv(f'{OUTPUT_DIR}/CSV/transformer_results_D1.csv', index=False)
print('Saved: transformer_results_D1.csv')


# %% [code] (Cell 101)
# Transformer confusion matrix
cm_trans = confusion_matrix(y_test, y_trans)
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
sns.heatmap(cm_trans, annot=True, fmt='d', cmap='Oranges', ax=axes[0],
            xticklabels=label_names, yticklabels=label_names)
axes[0].set_title('Twitter-RoBERTa — Confusion Matrix', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Predicted'); axes[0].set_ylabel('Actual')
cm_tn = cm_trans.astype(float) / cm_trans.sum(axis=1, keepdims=True)
sns.heatmap(cm_tn, annot=True, fmt='.2f', cmap='Oranges', ax=axes[1],
            xticklabels=label_names, yticklabels=label_names)
axes[1].set_title('Twitter-RoBERTa — Normalized', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Predicted'); axes[1].set_ylabel('Actual')
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/figures/transformer_confusion_matrix.png', dpi=150, bbox_inches='tight')
plt.show()

# Add to all_results for final comparison
all_results['Twitter-RoBERTa'] = {
    'Accuracy': trans_acc, 'Macro Precision': trans_prec, 'Macro Recall': trans_rec,
    'Macro F1': trans_mf1, 'Weighted F1': trans_wf1,
    'Training Time': trans_train_time, 'Inference Time': trans_inf_time
}
y_trans_stored = y_trans.copy()
print('Twitter-RoBERTa added to model registry.')


# %% [markdown]
# ## Advanced Experiment H — Twitter-RoBERTa Multi-Seed Uncertainty
# Running seeds 42, 123, 2026 sequentially. Seed 42 already run above — reusing those results.


# %% [code] (Cell 103)
trans_ms_rows = [{
    'Seed': 42,
    'Test Macro F1': round(trans_mf1, 4),
    'Test Weighted F1': round(trans_wf1, 4),
    'Test Accuracy': round(trans_acc, 4),
    'Training Time (s)': round(trans_train_time, 1),
    'Inference Time (s)': round(trans_inf_time, 2),
    'Parameters': trans_param_count
}]

for extra_seed in [123, 2026]:
    print(f'\n--- Twitter-RoBERTa Seed {extra_seed} ---')
    try:
        hf_set_seed(extra_seed)
        ms_trans_model = AutoModelForSequenceClassification.from_pretrained(
            TF_MODEL_ID, num_labels=NUM_CLASSES,
            id2label=TRANS_ID2LABEL, label2id=TRANS_LABEL2ID,
            ignore_mismatched_sizes=True
        )
        ms_ckpt_dir = f'{OUTPUT_DIR}/models/transformer_checkpoint_s{extra_seed}'
        ms_args = TrainingArguments(
            output_dir=ms_ckpt_dir,
            num_train_epochs=TRANS_EPOCHS,
            per_device_train_batch_size=TRANS_BATCH,
            per_device_eval_batch_size=TRANS_BATCH * 2,
            learning_rate=TRANS_LR, weight_decay=TRANS_WD,
            evaluation_strategy='epoch', save_strategy='epoch',
            load_best_model_at_end=True, metric_for_best_model='macro_f1',
            seed=extra_seed, fp16=(TRANS_DEVICE=='cuda'),
            logging_steps=50, report_to='none', save_total_limit=1
        )
        ms_trainer = Trainer(
            model=ms_trans_model, args=ms_args,
            train_dataset=train_ds, eval_dataset=val_ds,
            tokenizer=hf_tokenizer, compute_metrics=compute_metrics_hf,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        t0 = time.time()
        ms_trainer.train()
        ms_train_t = time.time() - t0
        t1 = time.time()
        ms_preds = ms_trainer.predict(test_ds)
        ms_inf_t = time.time() - t1
        y_ms_trans = np.argmax(ms_preds.predictions, axis=1)
        ms_mf1 = f1_score(y_test, y_ms_trans, average='macro', zero_division=0)
        ms_wf1 = f1_score(y_test, y_ms_trans, average='weighted', zero_division=0)
        ms_acc = accuracy_score(y_test, y_ms_trans)
        trans_ms_rows.append({
            'Seed': extra_seed,
            'Test Macro F1': round(ms_mf1, 4),
            'Test Weighted F1': round(ms_wf1, 4),
            'Test Accuracy': round(ms_acc, 4),
            'Training Time (s)': round(ms_train_t, 1),
            'Inference Time (s)': round(ms_inf_t, 2),
            'Parameters': trans_param_count
        })
        print(f'  Seed {extra_seed}: Test Macro F1={ms_mf1:.4f}')
        del ms_trans_model, ms_trainer
        if TRANS_DEVICE == 'cuda': torch.cuda.empty_cache()
    except Exception as e:
        print(f'  Seed {extra_seed} FAILED: {e}')
        trans_ms_rows.append({'Seed': extra_seed, 'Test Macro F1': 'NOT EXECUTED',
                              'Test Weighted F1': 'NOT EXECUTED', 'Test Accuracy': 'NOT EXECUTED',
                              'Training Time (s)': 'NOT EXECUTED', 'Inference Time (s)': 'NOT EXECUTED',
                              'Parameters': trans_param_count, 'Reason': str(e)})

trans_ms_df = pd.DataFrame(trans_ms_rows)
trans_ms_df.to_csv(f'{OUTPUT_DIR}/CSV/transformer_multiseed_results_D1.csv', index=False)
print('Saved: transformer_multiseed_results_D1.csv')
display(trans_ms_df)

# Summary stats for completed seeds
completed = trans_ms_df[trans_ms_df['Test Macro F1'] != 'NOT EXECUTED'].copy()
if len(completed) > 0:
    mf1_tr = completed['Test Macro F1'].astype(float).values
    print(f'\nCompleted seeds: {completed["Seed"].tolist()}')
    print(f'Mean Macro F1: {mf1_tr.mean():.4f}  |  SD: {mf1_tr.std():.4f}')
    if len(mf1_tr) > 1:
        ci = stats.t.interval(0.95, df=len(mf1_tr)-1, loc=mf1_tr.mean(), scale=stats.sem(mf1_tr))
        print(f'95% CI: ({ci[0]:.4f}, {ci[1]:.4f})')


# %% [markdown]
# ## Advanced Experiment I — Cross-Dataset Transfer
# D1 and D2 share compatible labels: `negative / neutral / positive`.  
# D2 (US Airline) is loaded here temporarily for transfer experiments. Existing D1 within-dataset results are unchanged.


# %% [code] (Cell 105)
# Label harmonization documentation
label_harmonization = pd.DataFrame([
    {'Dataset': 'D1 (tweet_eval sentiment)', 'Label_0': 'negative', 'Label_1': 'neutral', 'Label_2': 'positive',
     'Source': 'cardiffnlp/tweet_eval', 'Compatible': True},
    {'Dataset': 'D2 (US Airline Twitter)',   'Label_0': 'negative', 'Label_1': 'neutral', 'Label_2': 'positive',
     'Source': 'Tweets.csv', 'Compatible': True},
])
label_harmonization.to_csv(f'{OUTPUT_DIR}/CSV/label_harmonization.csv', index=False)
print('Label mapping is IDENTICAL across both datasets:')
print('  negative → negative (0)')
print('  neutral  → neutral  (1)')
print('  positive → positive (2)')
print('No remapping required.')
display(label_harmonization)


# %% [code] (Cell 106)
# Load D2 for cross-dataset transfer
from google.colab import files as colab_files
print('Upload Tweets.csv (Dataset 2) for cross-dataset transfer:')
d2_uploaded = colab_files.upload()
d2_fname = list(d2_uploaded.keys())[0]
d2_df_full = pd.read_csv(d2_fname)
print(f'D2 loaded: {d2_df_full.shape}')


# %% [code] (Cell 107)
from sklearn.preprocessing import LabelEncoder as LE2

# Prepare D2 data
d2_df = d2_df_full[['text','airline_sentiment']].dropna().copy()
d2_df['clean_text'] = d2_df['text'].apply(clean_tweet)
d2_df = d2_df[d2_df['clean_text'].str.len() > 0].reset_index(drop=True)

le2 = LE2()
le2.fit(['negative','neutral','positive'])
d2_df['label_enc'] = le2.transform(d2_df['airline_sentiment'])

# Stratified train/test split for D2 (70/30)
X_d2, y_d2 = d2_df['clean_text'].values, d2_df['label_enc'].values
from sklearn.model_selection import train_test_split as tts
X_d2_train, X_d2_test, y_d2_train, y_d2_test = tts(
    X_d2, y_d2, test_size=0.30, stratify=y_d2, random_state=RANDOM_SEED
)
print(f'D2 train: {len(X_d2_train)}  |  D2 test: {len(X_d2_test)}')


# %% [code] (Cell 108)
# Within-dataset baselines (already computed above)
d1_within_mf1 = f1_score(y_test, y_svc, average='macro', zero_division=0)
d1_within_wf1 = f1_score(y_test, y_svc, average='weighted', zero_division=0)
d1_within_acc = accuracy_score(y_test, y_svc)

# Experiment A: Train on D1, Test on D2
cross_pipe_D1toD2 = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000, sublinear_tf=True, min_df=2)),
    ('clf', LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_SEED))
])
cross_pipe_D1toD2.fit(X_train, y_train)
y_cross_D1toD2 = cross_pipe_D1toD2.predict(X_d2_test)
d1tod2_mf1 = f1_score(y_d2_test, y_cross_D1toD2, average='macro',    zero_division=0)
d1tod2_wf1 = f1_score(y_d2_test, y_cross_D1toD2, average='weighted', zero_division=0)
d1tod2_acc = accuracy_score(y_d2_test, y_cross_D1toD2)

# Experiment B: Train on D2, Test on D1
cross_pipe_D2toD1 = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000, sublinear_tf=True, min_df=2)),
    ('clf', LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_SEED))
])
cross_pipe_D2toD1.fit(X_d2_train, y_d2_train)
y_cross_D2toD1 = cross_pipe_D2toD1.predict(X_test)
d2tod1_mf1 = f1_score(y_test,    y_cross_D2toD1, average='macro',    zero_division=0)
d2tod1_wf1 = f1_score(y_test,    y_cross_D2toD1, average='weighted', zero_division=0)
d2tod1_acc = accuracy_score(y_test, y_cross_D2toD1)

# D2 within-dataset baseline
d2_within_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(ngram_range=(1,2), max_features=50000, sublinear_tf=True, min_df=2)),
    ('clf', LinearSVC(C=0.5, max_iter=2000, random_state=RANDOM_SEED))
])
d2_within_pipe.fit(X_d2_train, y_d2_train)
y_d2_within = d2_within_pipe.predict(X_d2_test)
d2_within_mf1 = f1_score(y_d2_test, y_d2_within, average='macro',    zero_division=0)
d2_within_wf1 = f1_score(y_d2_test, y_d2_within, average='weighted', zero_division=0)
d2_within_acc = accuracy_score(y_d2_test, y_d2_within)

transfer_df = pd.DataFrame([
    {'Source': 'D1 (tweet_eval)', 'Target': 'D1 (within)', 'Model': 'LinearSVC',
     'Macro F1': round(d1_within_mf1,4), 'Weighted F1': round(d1_within_wf1,4), 'Accuracy': round(d1_within_acc,4),
     'Drop vs Within': 0.0},
    {'Source': 'D1 (tweet_eval)', 'Target': 'D2 (airline)', 'Model': 'LinearSVC',
     'Macro F1': round(d1tod2_mf1,4), 'Weighted F1': round(d1tod2_wf1,4), 'Accuracy': round(d1tod2_acc,4),
     'Drop vs Within': round(d1_within_mf1 - d1tod2_mf1, 4)},
    {'Source': 'D2 (airline)',   'Target': 'D2 (within)', 'Model': 'LinearSVC',
     'Macro F1': round(d2_within_mf1,4), 'Weighted F1': round(d2_within_wf1,4), 'Accuracy': round(d2_within_acc,4),
     'Drop vs Within': 0.0},
    {'Source': 'D2 (airline)',   'Target': 'D1 (tweet_eval)', 'Model': 'LinearSVC',
     'Macro F1': round(d2tod1_mf1,4), 'Weighted F1': round(d2tod1_wf1,4), 'Accuracy': round(d2tod1_acc,4),
     'Drop vs Within': round(d2_within_mf1 - d2tod1_mf1, 4)},
])
print('Cross-Dataset Transfer Results:')
display(transfer_df)
transfer_df.to_csv(f'{OUTPUT_DIR}/CSV/cross_dataset_transfer_results.csv', index=False)
print('Saved: cross_dataset_transfer_results.csv')


# %% [markdown]
# ## Advanced Experiment J — Fixed Challenge Set
# Curating challenge examples from the **test set** by pattern matching. Labels are taken directly from the dataset — no fabrication.


# %% [code] (Cell 110)
import unicodedata

def has_emoji(text):
    return any(unicodedata.category(c) in ('So','Sm') or ord(c) > 0x1F300 for c in str(text))

def challenge_type(text):
    t = str(text).lower()
    types = []
    if re.search(r"\b(not|never|no|n't|neither|nor|without)\b", t): types.append('negation')
    if len(re.findall(r'#\w+', t)) >= 2: types.append('hashtag_heavy')
    if has_emoji(text): types.append('emoji_heavy')
    if len(t.split()) < 5: types.append('short_ambiguous')
    if re.search(r'\b(lol|omg|smh|tbh|imo|imho|fwiw|btw|af|irl|irl|yolo|fomo)\b', t): types.append('slang')
    if not types: types.append('other')
    return '|'.join(types)

# Build challenge set from test data
challenge_records = []
for i, (txt, lbl) in enumerate(zip(X_test, y_test)):
    ctype = challenge_type(txt)
    if ctype != 'other':
        challenge_records.append({
            'row_id': i, 'text': txt,
            'true_label': label_names[lbl],
            'true_label_enc': lbl,
            'challenge_type': ctype,
            'dataset': 'D1'
        })

chal_df = pd.DataFrame(challenge_records)
print(f'Challenge set size: {len(chal_df)}')
print('By type:')
type_counts = {}
for row in challenge_records:
    for t in row['challenge_type'].split('|'):
        type_counts[t] = type_counts.get(t, 0) + 1
for t, c in sorted(type_counts.items(), key=lambda x:-x[1]):
    print(f'  {t}: {c}')
chal_df.to_csv(f'{OUTPUT_DIR}/CSV/challenge_set_D1.csv', index=False)
print('Saved: challenge_set_D1.csv')


# %% [code] (Cell 111)
# Evaluate LinearSVC, BiLSTM, Transformer on challenge set
chal_idx = chal_df['row_id'].values
y_chal   = chal_df['true_label_enc'].values
X_chal_text = chal_df['text'].values

y_chal_svc   = svc_pipe.predict(X_chal_text)
y_chal_bilstm = np.argmax(bilstm.predict(X_test_seq[chal_idx], verbose=0), axis=1)

chal_results_rows = []
for ctype in list(type_counts.keys()):
    mask = chal_df['challenge_type'].str.contains(ctype)
    if mask.sum() == 0: continue
    y_c = y_chal[mask]
    row = {'Challenge Type': ctype, 'Count': int(mask.sum())}
    for model_name, preds in [('LinearSVC', y_chal_svc[mask]), ('BiLSTM', y_chal_bilstm[mask])]:
        row[f'{model_name} Accuracy'] = round(accuracy_score(y_c, preds), 4)
        row[f'{model_name} Macro F1'] = round(f1_score(y_c, preds, average='macro', zero_division=0), 4)
    chal_results_rows.append(row)

chal_results_df = pd.DataFrame(chal_results_rows)
print('Challenge Set Results by Type:')
display(chal_results_df)
chal_results_df.to_csv(f'{OUTPUT_DIR}/CSV/challenge_results_D1.csv', index=False)
print('Saved: challenge_results_D1.csv')


# %% [markdown]
# ## Advanced Experiment K — Temporal Holdout
# **Not applicable to Dataset 1.**  
# The `cardiffnlp/tweet_eval` sentiment subset does not expose individual tweet timestamps. The dataset provides pre-defined train/validation/test splits without temporal metadata. Temporal holdout was therefore not performed for Dataset 1.


# %% [markdown]
# ## Advanced Experiment L — Save/Reload Verification
# Verifying that saved models produce identical predictions when reloaded.


# %% [code] (Cell 114)
import json as json_lib

repro_results = {}
FIXED_SAMPLE_IDX = [0, 1, 2, 3, 4]
X_fixed = X_test[FIXED_SAMPLE_IDX]
y_fixed = y_test[FIXED_SAMPLE_IDX]

# 1. Classical pipeline
orig_pred_classical = best_classical_pipe.predict(X_fixed).tolist()
reloaded_pipe = joblib.load(f'{OUTPUT_DIR}/models/selected_classical_pipeline.joblib')
reload_pred_classical = reloaded_pipe.predict(X_fixed).tolist()
classical_match = orig_pred_classical == reload_pred_classical
repro_results['classical_pipeline'] = {
    'model': best_classical_name,
    'original_preds': orig_pred_classical,
    'reloaded_preds': reload_pred_classical,
    'match': classical_match
}
print(f'Classical ({best_classical_name}) reload: {"PASS" if classical_match else "FAIL"}')

# 2. BiLSTM
X_fixed_seq = X_test_seq[FIXED_SAMPLE_IDX]
orig_pred_bilstm = np.argmax(bilstm.predict(X_fixed_seq, verbose=0), axis=1).tolist()
reloaded_bilstm = tf.keras.models.load_model(f'{OUTPUT_DIR}/models/bilstm_model.keras')
reload_pred_bilstm = np.argmax(reloaded_bilstm.predict(X_fixed_seq, verbose=0), axis=1).tolist()
bilstm_match = orig_pred_bilstm == reload_pred_bilstm
repro_results['bilstm'] = {
    'original_preds': orig_pred_bilstm,
    'reloaded_preds': reload_pred_bilstm,
    'match': bilstm_match
}
print(f'BiLSTM reload: {"PASS" if bilstm_match else "FAIL"}')

# 3. Transformer
try:
    from transformers import pipeline as hf_pipeline
    reload_trans = hf_pipeline('text-classification', model=TRANS_CKPT_DIR,
                               tokenizer=hf_tokenizer, device=0 if TRANS_DEVICE=='cuda' else -1)
    fixed_texts = X_fixed.tolist()
    orig_trans_preds = [TRANS_LABEL2ID.get(r['label'].lower(), -1)
                        for r in reload_trans(fixed_texts, truncation=True, max_length=TRANS_MAX_LEN)]
    expected_trans = y_trans_stored[FIXED_SAMPLE_IDX].tolist()
    trans_match = orig_trans_preds == expected_trans
    repro_results['transformer'] = {
        'model': TF_MODEL_ID,
        'reloaded_preds': orig_trans_preds,
        'expected_preds': expected_trans,
        'match': trans_match
    }
    print(f'Transformer reload: {"PASS" if trans_match else "MISMATCH (check label format)"}')
except Exception as e:
    print(f'Transformer reload check skipped: {e}')
    repro_results['transformer'] = {'status': 'NOT EXECUTED', 'reason': str(e)}

repro_results['true_labels'] = y_fixed.tolist()
repro_results['texts'] = X_fixed.tolist()

with open(f'{OUTPUT_DIR}/metrics/reproducibility_check_D1.json', 'w') as jf:
    json_lib.dump(repro_results, jf, indent=2)
print('Saved: reproducibility_check_D1.json')


# %% [markdown]
# ## Advanced Experiment M — Final Advanced Model Comparison
# Comprehensive table including all models that actually ran in this notebook.


# %% [code] (Cell 116)
# Gather CV metrics
cv_lookup = {row['Model']: row for row in cv_rows}
char_cv_mf1 = char_cv_scores['test_macro_f1'].mean()
char_cv_sd  = char_cv_scores['test_macro_f1'].std()

# BiLSTM size info (already computed)
trans_size_mb = sum(
    os.path.getsize(os.path.join(dp, f))
    for dp, _, fnames in os.walk(TRANS_CKPT_DIR) for f in fnames
) / (1024**2) if os.path.exists(TRANS_CKPT_DIR) else float('nan')

adv_rows = []

def adv_row(model, representation, y_pred_test, cv_mf1=None, cv_sd=None,
            train_time=None, inf_time=None, n_params=None, size_mb=None):
    acc = accuracy_score(y_test, y_pred_test)
    mf1 = f1_score(y_test, y_pred_test, average='macro',    zero_division=0)
    wf1 = f1_score(y_test, y_pred_test, average='weighted', zero_division=0)
    return {
        'Model': model, 'Representation': representation,
        'Macro F1': round(mf1, 4), 'Weighted F1': round(wf1, 4), 'Accuracy': round(acc, 4),
        'CV Macro F1': round(cv_mf1, 4) if cv_mf1 is not None else 'N/A',
        'CV SD':       round(cv_sd,  4) if cv_sd  is not None else 'N/A',
        'Training Time': f'{train_time:.1f}s' if train_time else 'N/A',
        'Inference Latency': f'{inf_time:.3f}s' if inf_time else 'N/A',
        'Parameters': n_params if n_params else 'N/A',
        'Model Size (MB)': round(size_mb, 2) if size_mb and not (isinstance(size_mb, float) and size_mb != size_mb) else 'N/A'
    }

adv_rows.append(adv_row('Logistic Regression', 'Word TF-IDF (1-2gram)', y_lr,
    cv_mf1=cv_lookup.get('Logistic Regression',{}).get('CV Macro F1 Mean'),
    cv_sd=cv_lookup.get('Logistic Regression',{}).get('CV Macro F1 SD'),
    train_time=try_float(all_results.get('Logistic Regression',{}).get('Training Time')),
    inf_time=try_float(all_results.get('Logistic Regression',{}).get('Inference Time'))))

adv_rows.append(adv_row('LinearSVC', 'Word TF-IDF (1-2gram)', y_svc,
    cv_mf1=cv_lookup.get('LinearSVC',{}).get('CV Macro F1 Mean'),
    cv_sd=cv_lookup.get('LinearSVC',{}).get('CV Macro F1 SD'),
    train_time=try_float(all_results.get('LinearSVC',{}).get('Training Time')),
    inf_time=try_float(all_results.get('LinearSVC',{}).get('Inference Time'))))

adv_rows.append(adv_row('Char TF-IDF + LinearSVC', 'Char TF-IDF (2-5gram)', y_char,
    cv_mf1=char_cv_mf1, cv_sd=char_cv_sd,
    train_time=char_train_time, inf_time=char_inf_time))

adv_rows.append(adv_row('BiLSTM', 'Embedding + BiLSTM', y_bilstm,
    train_time=bilstm_train_time, inf_time=bilstm_inf_time,
    n_params=bilstm_total_params, size_mb=model_size_mb))

try:
    adv_rows.append(adv_row('Twitter-RoBERTa', 'Transformer (125M)', y_trans,
        train_time=trans_train_time, inf_time=trans_inf_time,
        n_params=trans_param_count, size_mb=trans_size_mb))
except NameError:
    adv_rows.append({'Model': 'Twitter-RoBERTa', 'Representation': 'Transformer (125M)',
                     'Macro F1': 'Not executed', 'Weighted F1': 'Not executed',
                     'Accuracy': 'Not executed', 'CV Macro F1': 'N/A', 'CV SD': 'N/A',
                     'Training Time': 'N/A', 'Inference Latency': 'N/A',
                     'Parameters': 'N/A', 'Model Size (MB)': 'N/A'})

adv_df = pd.DataFrame(adv_rows)
adv_df.to_csv(f'{OUTPUT_DIR}/CSV/final_model_comparison_D1.csv', index=False)
print('Final Advanced Model Comparison — Dataset 1')
display(adv_df)
print('Saved: final_model_comparison_D1.csv')


# %% [code] (Cell 117)
# Bar chart — final comparison by Macro F1
def safe_float(x):
    try: return float(x)
    except: return None

plot_rows = [(r['Model'], safe_float(r['Macro F1'])) for _, r in adv_df.iterrows() if safe_float(r['Macro F1']) is not None]
if plot_rows:
    p_models, p_vals = zip(*plot_rows)
    colors_adv = ['#8e44ad' if 'RoBERTa' in m else ('#2c3e50' if 'LSTM' in m else '#2980b9') for m in p_models]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(p_models, p_vals, color=colors_adv, edgecolor='white', alpha=0.9)
    for bar, val in zip(bars, p_vals):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.005,
                f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')
    ax.set_title('Final Advanced Model Comparison — Macro F1 (Dataset 1)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Macro F1')
    ax.set_ylim(0, min(1.15, max(p_vals)+0.15))
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.xticks(rotation=15, ha='right')
    plt.tight_layout()
    plt.savefig(f'{OUTPUT_DIR}/figures/final_advanced_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    print('Saved: final_advanced_comparison.png')


# %% [markdown]
# ## Advanced Experiments Added — Dataset 1
# The following checklist is generated at runtime to reflect what actually executed.


# %% [code] (Cell 119)
checklist_items = [
    ('5-Fold Cross-Validation',       'cv_results_D1.csv'),
    ('Character TF-IDF',               'character_tfidf_results_D1.csv'),
    ('BiLSTM Multi-Seed Uncertainty',  'bilstm_multiseed_results_D1.csv'),
    ('BiLSTM Model Size & Parameters', 'bilstm_size_info_D1.csv'),
    ('Confidence Analysis',            'calibration_plot.png'),
    ('Confidence-Based Abstention',    'abstention_results_D1.csv'),
    ('Twitter-RoBERTa Fine-tuning',    'transformer_results_D1.csv'),
    ('Transformer Multi-Seed',         'transformer_multiseed_results_D1.csv'),
    ('Cross-Dataset Transfer',         'cross_dataset_transfer_results.csv'),
    ('Challenge Set Evaluation',       'challenge_results_D1.csv'),
    ('Temporal Holdout',               None),
    ('Save/Reload Verification',       'reproducibility_check_D1.json'),
    ('Final Advanced Comparison',      'final_model_comparison_D1.csv'),
]

print('Advanced Experiments Checklist — Dataset 1')
print('=' * 55)
for name, artifact in checklist_items:
    if name == 'Temporal Holdout':
        print(f'  [NOTE] {name}: No timestamps in tweet_eval sentiment subset.')
        continue
    # Check CSV
    found = False
    for sub in ['CSV','figures','metrics','models']:
        path = os.path.join(OUTPUT_DIR, sub, artifact) if artifact else ''
        if path and os.path.exists(path):
            found = True; break
    status = 'DONE' if found else 'NOT FOUND'
    mark = 'OK' if found else 'XX'
    print(f'  [{mark}] {name}: {status}')
print('=' * 55)


# %% [markdown]
# ## 31. Output File Inventory


# %% [code] (Cell 121)
print('Saved Output Files')
print('-' * 60)
for root, dirs, files in os.walk(OUTPUT_DIR):
    dirs.sort()
    for fname in sorted(files):
        fp = os.path.join(root, fname)
        sz = os.path.getsize(fp) / 1024
        print(f'  {fp}  ({sz:.1f} KB)')

