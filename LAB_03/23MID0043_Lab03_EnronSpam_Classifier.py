"""Auto-generated Python script from 23MID0043_Lab03_EnronSpam_Classifier.ipynb."""

# %%
# Run this on Google Colab if any package is missing
# !pip install -q pandas numpy scikit-learn matplotlib seaborn joblib tensorflow openai python-dotenv

# %%
# Upload dataset in Colab
try:
    from google.colab import files
    uploaded = files.upload()
    print('Uploaded:', list(uploaded.keys()))
except ImportError:
    print('Running locally; ensure the CSV is in the working directory.')

# %%
import io
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_auc_score, roc_curve, ConfusionMatrixDisplay
)
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

OUT_DIR = Path('outputs')
OUT_DIR.mkdir(exist_ok=True)

# %%
def load_data():
    try:
        from google.colab import files  # noqa: F401
        buf = io.BytesIO(uploaded['enron_spam_data.csv'])
        df = pd.read_csv(buf)
    except Exception:
        df = pd.read_csv('enron_spam_data.csv')

    df['label'] = df['Spam/Ham'].str.lower().map({'spam': 1, 'ham': 0})
    df['Subject'] = df['Subject'].fillna('')
    df['Message'] = df['Message'].fillna('')
    df['message'] = (df['Subject'] + ' ' + df['Message']).astype(str)
    df = df[['message', 'label']].copy()
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    return df

df = load_data()
print('Shape:', df.shape)
print('Label distribution:')
print(df['label'].value_counts())
df.head()

# %%
print('Info:')
df.info()
print('Missing values:', df.isnull().sum().to_dict())
print('Class proportions:')
print(df['label'].value_counts(normalize=True))

# %%
df['msg_len'] = df['message'].apply(len)
plt.figure(figsize=(10, 4))
sns.histplot(data=df, x='msg_len', hue='label', bins=50, kde=True)
plt.title('Message Length Distribution by Label')
plt.xlabel('Characters')
plt.xlim(0, df['msg_len'].quantile(0.99))
plt.show()

# %%
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+|www\.\S+', ' ', text)
    text = re.sub(r'<[^>]+>', ' ', text)
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df['clean_message'] = df['message'].apply(clean_text)
df = df[df['clean_message'].str.len() > 0].copy()
print('Shape after cleaning:', df.shape)
df[['message', 'clean_message', 'label']].head()

# %%
X = df['clean_message']
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
)
print('Train:', X_train.shape, 'Test:', X_test.shape)
print('Train label distribution:')
print(y_train.value_counts(normalize=True))

# %%
models = {
    'Dummy': DummyClassifier(strategy='most_frequent'),
    'MultinomialNB': MultinomialNB(),
    'ComplementNB': ComplementNB(),
    'LogisticRegression': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    'LinearSVC': LinearSVC(random_state=RANDOM_STATE)
}

results = []
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

for name, model in models.items():
    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))),
        ('clf', model)
    ])
    scores = cross_validate(
        pipe, X_train, y_train, cv=cv,
        scoring=['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
    )
    results.append({
        'Model': name,
        'Accuracy': scores['test_accuracy'].mean(),
        'F1': scores['test_f1'].mean(),
        'Precision': scores['test_precision'].mean(),
        'Recall': scores['test_recall'].mean(),
        'ROC_AUC': scores['test_roc_auc'].mean()
    })

results_df = pd.DataFrame(results).sort_values('F1', ascending=False)
display(results_df)

# %%
best_name = results_df.iloc[0]['Model']
print('Best TF-IDF model:', best_name)

best_pipe = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', max_features=10000, ngram_range=(1, 2))),
    ('clf', models[best_name])
])
best_pipe.fit(X_train, y_train)
y_pred = best_pipe.predict(X_test)

print('Accuracy:', accuracy_score(y_test, y_pred))
print('F1:', f1_score(y_test, y_pred))
print('ROC AUC:', roc_auc_score(y_test, y_pred))
print('Classification Report:')
print(classification_report(y_test, y_pred, digits=4))

# %%
fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax[0], cmap='Blues')
ax[0].set_title('Confusion Matrix')

if hasattr(best_pipe.named_steps['clf'], 'decision_function'):
    scores = best_pipe.decision_function(X_test)
else:
    scores = best_pipe.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, scores)
ax[1].plot(fpr, tpr, label=f'AUC = {roc_auc_score(y_test, scores):.4f}')
ax[1].plot([0, 1], [0, 1], 'k--')
ax[1].set_xlabel('False Positive Rate')
ax[1].set_ylabel('True Positive Rate')
ax[1].set_title('ROC Curve')
ax[1].legend()
plt.tight_layout()
plt.show()

# %%
joblib.dump(best_pipe, OUT_DIR / 'enron_best_tfidf.pkl')
print('Saved:', OUT_DIR / 'enron_best_tfidf.pkl')

# %%
sample_email = """
Subject: Quarterly budget review meeting

Hi team, please find the attached slides for tomorrow's quarterly budget review. Let me know if you have any questions.
"""

clean_sample = clean_text(sample_email)
pred = best_pipe.predict([clean_sample])[0]
print('Predicted label:', pred, '(1=spam, 0=ham)')

# %%
MAX_WORDS = 20000
MAX_LEN = 200
EMBED_DIM = 128

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = pad_sequences(tokenizer.texts_to_sequences(X_train), maxlen=MAX_LEN)
X_test_seq = pad_sequences(tokenizer.texts_to_sequences(X_test), maxlen=MAX_LEN)

bilstm_model = Sequential([
    Embedding(MAX_WORDS, EMBED_DIM, input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(le.classes_), activation='softmax')
])

bilstm_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = bilstm_model.fit(
    X_train_seq, y_train_enc,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early],
    verbose=1
)

# %%
loss, acc = bilstm_model.evaluate(X_test_seq, y_test_enc, verbose=0)
y_proba = bilstm_model.predict(X_test_seq, verbose=0)
y_pred_bilstm = y_proba.argmax(axis=1)

print('BiLSTM Accuracy:', acc)
print('BiLSTM F1:', f1_score(y_test_enc, y_pred_bilstm))
print('BiLSTM ROC AUC:', roc_auc_score(y_test_enc, y_proba[:, 1]))
print('Classification Report:')
print(classification_report(y_test_enc, y_pred_bilstm, target_names=[str(c) for c in le.classes_], digits=4))

# %%
bilstm_model.save(OUT_DIR / 'enron_bilstm.keras')
print('Saved:', OUT_DIR / 'enron_bilstm.keras')

# %%
sample_seq = pad_sequences(tokenizer.texts_to_sequences([clean_sample]), maxlen=MAX_LEN)
pred_bilstm = bilstm_model.predict(sample_seq, verbose=0)
label_bilstm = le.inverse_transform([pred_bilstm.argmax()])[0]
print('BiLSTM predicted label:', label_bilstm, '(1=spam, 0=ham)')

# %%
import os
from openai import OpenAI

api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    client = OpenAI(api_key=api_key)

    def generate_reply(email_text, predicted_class):
        prompt = f"""
You are a professional email assistant.
Predicted Category: {predicted_class} (0=ham, 1=spam)
Email:
{email_text}

If the email is ham, generate a concise, professional reply draft.
If the email is spam, generate a brief warning note explaining why it looks suspicious.
"""
        response = client.chat.completions.create(
            model=os.getenv('OPENAI_MODEL', 'gpt-3.5-turbo'),
            messages=[{'role': 'user', 'content': prompt}]
        )
        return response.choices[0].message.content

    draft = generate_reply(sample_email, pred)
    print(draft)
else:
    print('Set OPENAI_API_KEY as an environment variable to enable LLM draft generation.')
