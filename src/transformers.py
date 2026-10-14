import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class MultiHotEncoder(BaseEstimator, TransformerMixin):
    """Turn ';'-separated multi-select columns into 0/1 columns.
    The vocabulary is learned on training data only."""

    def __init__(self, min_freq=0.01, sep=";"):
        self.min_freq = min_freq
        self.sep = sep

    def _tokens(self, series):
        return series.fillna("").astype(str).apply(
            lambda s: {t.strip() for t in s.split(self.sep) if t.strip()}
        )

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.vocab_ = {}
        for col in X.columns:
            token_sets = self._tokens(X[col])
            flat = pd.Series([t for s in token_sets for t in s], dtype=object)
            counts = flat.value_counts()
            keep = counts[counts >= self.min_freq * len(X)].index.tolist()
            self.vocab_[col] = sorted(keep)
        self.feature_names_ = [f"{col}={tok}"
                               for col, toks in self.vocab_.items() for tok in toks]
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        blocks = []
        for col, toks in self.vocab_.items():
            token_sets = self._tokens(X[col])
            blocks.append(np.array([[int(t in s) for t in toks] for s in token_sets],
                                   dtype=np.float32))
        return np.hstack(blocks)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_)