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

class SkillCounts(BaseEstimator, TransformerMixin):
    """Number of answers ticked in each multi-select column (breadth of skills).
    Nothing is learned, so fit only records the column names."""

    def __init__(self, sep=";"):
        self.sep = sep

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X)
        return np.column_stack([
            X[c].fillna("").astype(str).apply(
                lambda s: sum(1 for t in s.split(self.sep) if t.strip()))
            for c in self.columns_]).astype(np.float32)

    def get_feature_names_out(self, input_features=None):
        return np.array([f"n_{c}" for c in self.columns_])
