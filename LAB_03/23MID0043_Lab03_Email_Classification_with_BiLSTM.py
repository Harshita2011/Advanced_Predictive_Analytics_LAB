"""Auto-generated Python script from 23MID0043_Lab03_Email_Classification_with_BiLSTM.ipynb."""

# %%
# Install packages (Colab)
# !pip install -q pandas numpy scipy scikit-learn matplotlib joblib openai python-dotenv sentence-transformers tensorflow

# %%
from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.dummy import DummyClassifier
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

RANDOM_STATE=42
DATA_DIR=Path("data")
OUT_DIR=Path("outputs")
OUT_DIR.mkdir(exist_ok=True)

# %%
DATASETS={
"business_intent":"data/business_email_intent.csv",
"enron_spam":"data/enron_spam.csv",
"spamassassin":"data/spamassassin.csv"
}

def load_dataset(path):
    df=pd.read_csv(path)
    df["subject"]=df["subject"].fillna("")
    df["body"]=df["body"].fillna("")
    df["text"]="subject: "+df["subject"]+"\nbody: "+df["body"]
    return df

# %%
dataset_name="business_intent"
df=load_dataset(DATASETS[dataset_name])

display(df.head())
print(df.label.value_counts())

# %%
X=df["text"]
y=df["label"]

X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=0.2,stratify=y,random_state=RANDOM_STATE)

# %%
models={
"Dummy":DummyClassifier(strategy="most_frequent"),
"MultinomialNB":MultinomialNB(),
"ComplementNB":ComplementNB(),
"LogisticRegression":LogisticRegression(max_iter=1000),
"LinearSVC":LinearSVC()
}

results=[]

for name,model in models.items():
    pipe=Pipeline([
        ("tfidf",TfidfVectorizer(stop_words="english")),
        ("clf",model)
    ])

    cv=StratifiedKFold(n_splits=5,shuffle=True,random_state=RANDOM_STATE)

    scores=cross_validate(
        pipe,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy","f1_macro"]
    )

    results.append({
        "Model":name,
        "Accuracy":scores["test_accuracy"].mean(),
        "MacroF1":scores["test_f1_macro"].mean()
    })

results=pd.DataFrame(results).sort_values("MacroF1",ascending=False)
display(results)

# %%
best_model=results.iloc[0]["Model"]

pipe=Pipeline([
("tfidf",TfidfVectorizer(stop_words="english")),
("clf",models[best_model])
])

pipe.fit(X_train,y_train)

pred=pipe.predict(X_test)

print("Accuracy:",accuracy_score(y_test,pred))
print("Macro F1:",f1_score(y_test,pred,average="macro"))
print(classification_report(y_test,pred))

cm=confusion_matrix(y_test,pred)
print(cm)

# %%
joblib.dump(pipe,"outputs/best_email_classifier.pkl")
print("Model Saved")

# %%
sample_email="""
Subject: Meeting tomorrow

Can we reschedule tomorrow's project discussion to 3 PM?
"""

prediction=pipe.predict([sample_email])[0]
print("Predicted Class:",prediction)

# %%
# -------- LLM Draft Generation --------
from openai import OpenAI

client=OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_reply(email_text,predicted_class):
    prompt=f'''
You are a professional email assistant.

Predicted Category: {predicted_class}

Email:
{email_text}

Generate only a professional reply draft.
'''

    response=client.responses.create(
        model=os.getenv("OPENAI_MODEL","gpt-5-mini"),
        input=prompt
    )

    return response.output_text

# Uncomment after setting API Key
# draft=generate_reply(sample_email,prediction)
# print(draft)

# %%
# TensorFlow / Keras BiLSTM
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

MAX_WORDS=20000
MAX_LEN=200
EMBED_DIM=128

le=LabelEncoder()
y_train_enc=le.fit_transform(y_train)
y_test_enc=le.transform(y_test)

tokenizer=Tokenizer(num_words=MAX_WORDS,oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq=pad_sequences(tokenizer.texts_to_sequences(X_train),maxlen=MAX_LEN)
X_test_seq=pad_sequences(tokenizer.texts_to_sequences(X_test),maxlen=MAX_LEN)

model=Sequential([
    Embedding(MAX_WORDS,EMBED_DIM,input_length=MAX_LEN),
    Bidirectional(LSTM(64)),
    Dropout(0.5),
    Dense(64,activation="relu"),
    Dense(len(le.classes_),activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

early=EarlyStopping(
    monitor="val_loss",
    patience=3,
    restore_best_weights=True
)

history=model.fit(
    X_train_seq,
    y_train_enc,
    validation_split=0.2,
    epochs=10,
    batch_size=32,
    callbacks=[early],
    verbose=1
)

loss,acc=model.evaluate(X_test_seq,y_test_enc)
print("BiLSTM Accuracy:",acc)

pred=model.predict(X_test_seq)
pred_labels=pred.argmax(axis=1)

from sklearn.metrics import classification_report,f1_score

print("Macro F1:",f1_score(y_test_enc,pred_labels,average="macro"))
print(classification_report(y_test_enc,pred_labels,target_names=le.classes_))

model.save("outputs/bilstm_email_classifier.keras")
print("BiLSTM model saved.")

# %%
# Predict using BiLSTM

sample_seq=pad_sequences(
    tokenizer.texts_to_sequences([sample_email]),
    maxlen=MAX_LEN
)

pred=model.predict(sample_seq)
label=le.inverse_transform([pred.argmax()])[0]
print("BiLSTM Prediction:",label)
