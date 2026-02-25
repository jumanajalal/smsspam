import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV, cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from features import CustomFeatures
from utils import normalize_text

param_grid = {
    "clf__estimator__C": [0.1, 1, 10],
}

# Load dataset
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None)
df.columns = ["label", "message"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]

inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

word_tfidf = TfidfVectorizer(
    analyzer="word",
    stop_words="english",
    ngram_range=(1,2),
    preprocessor=normalize_text
)

char_tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5),
    preprocessor=normalize_text
)

import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

# Combine both
combined_features = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf),
    ("custom", CustomFeatures())
])

# Build base pipeline
pipeline = Pipeline([
    ("features", combined_features),
    ("clf", CalibratedClassifierCV(LinearSVC(max_iter=10000)))
])

# Inner CV (hyperparameter tuning)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=3,
    cv=inner_cv,
    scoring="f1",
    random_state=42,
    n_jobs=-1
)

# Outer CV (true evaluation)
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

nested_scores = cross_val_score(
    search,
    X,
    y,
    cv=outer_cv,
    scoring="f1"
)

print("Nested CV F1 Score:", nested_scores.mean())

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

search.fit(X_train, y_train)
best_model = search.best_estimator_

probs = best_model.predict_proba(X_val)[:, 1]

import numpy as np
from sklearn.metrics import f1_score

# Get probabilities ONLY on validation set
probs = best_model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.1, 0.9, 0.01)
best_threshold = 0.5
best_f1 = 0

for t in thresholds:
    preds = (probs >= t).astype(int)
    score = f1_score(y_val, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = t

print("Best Threshold:", best_threshold)
print("Best Validation F1 Score:", best_f1)

# Get probabilities for validation set
val_probs = best_model.predict_proba(X_val)[:, 1]
# ROC Curve
fpr, tpr, _ = roc_curve(y_val, val_probs)
roc_auc = roc_auc_score(y_val, val_probs)

plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0,1], [0,1], linestyle="--")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC)")
plt.legend()
plt.show(block=False)
plt.pause(9)
plt.close()


# Precision-Recall Curve
precision, recall, _ = precision_recall_curve(y_val, val_probs)
pr_auc = average_precision_score(y_val, val_probs)

plt.figure()
plt.plot(recall, precision, label=f"PR Curve (AP = {pr_auc:.3f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.show(block=False)
plt.pause(9)
plt.close()


import random

def obfuscate_text(text):
    replacements = {
        "a": "@",
        "e": "3",
        "i": "1",
        "o": "0",
        "s": "$"
    }

    # Randomly replace characters
    new_text = ""
    for char in text:
        if char.lower() in replacements and random.random() < 0.3:
            new_text += replacements[char.lower()]
        else:
            new_text += char

    # Random spacing attack
    if random.random() < 0.5:
        new_text = " ".join(list(new_text))

    return new_text

print("\n--- Adversarial Stress Test ---")

sample_spam = X_val[y_val == 1].sample(20, random_state=42)

original_preds = best_model.predict(sample_spam)
original_accuracy = (original_preds == 1).mean()

# Obfuscate messages
obfuscated = sample_spam.apply(obfuscate_text)

obf_preds = best_model.predict(obfuscated)
obf_accuracy = (obf_preds == 1).mean()

print("Original Spam Detection Rate:", round(original_accuracy, 3))
print("After Obfuscation Detection Rate:", round(obf_accuracy, 3))

# Save best model
with open("model.pkl", "wb") as f:
    pickle.dump({
        "model": best_model,
        "threshold": best_threshold
    }, f)

print("Model saved successfully.")
