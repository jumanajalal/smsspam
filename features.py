import re
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class CustomFeatures(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        features = []

        for text in X:
            has_url = 1 if re.search(r'http|www', text.lower()) else 0
            keyword_count = sum(word in text.lower() for word in ["free", "win", "cash", "click", "offer"])
            exclam_count = text.count("!")
            uppercase_ratio = sum(1 for c in text if c.isupper()) / (len(text)+1)

            features.append([has_url, keyword_count, exclam_count, uppercase_ratio])

        return np.array(features)