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

# Load dataset
df = pd.read_csv("SMSSpamCollection", sep="\t", header=None)
df.columns = ["label", "message"]

df["label"] = df["label"].map({"ham": 0, "spam": 1})

X = df["message"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Word-level TF-IDF
word_tfidf = TfidfVectorizer(
    analyzer="word",
    stop_words="english",
    ngram_range=(1,2)
)

# Character-level TF-IDF
char_tfidf = TfidfVectorizer(
    analyzer="char",
    ngram_range=(3,5)
)

# Combine both
combined_features = FeatureUnion([
    ("word", word_tfidf),
    ("char", char_tfidf)
])

# Build pipeline
pipeline = Pipeline([
    ("features", combined_features),
    ("clf", CalibratedClassifierCV(LinearSVC()))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
with open("model.pkl", "wb") as f:
    pickle.dump(pipeline, f)

print("Model saved successfully.")
