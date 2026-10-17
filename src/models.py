import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone

from src.evaluation import TIER_ORDER


class OrdinalClassifier(BaseEstimator, ClassifierMixin):
    """Frank & Hall ordinal classifier: one binary model per threshold
    (is the tier above Low? above Mid?), combined into Low/Mid/High probabilities."""

    def __init__(self, estimator, order=tuple(TIER_ORDER)):
        self.estimator = estimator
        self.order = order

    def fit(self, X, y):
        rank = np.asarray([self.order.index(v) for v in np.asarray(y).astype(str)])
        self.models_ = [clone(self.estimator).fit(X, (rank > k).astype(int))
                        for k in range(len(self.order) - 1)]
        self.classes_ = np.array(self.order)
        return self

    def predict_proba(self, X):
        above = np.column_stack([m.predict_proba(X)[:, 1] for m in self.models_])
        # keep P(above Mid) <= P(above Low) so no probability is negative
        above[:, 1] = np.minimum(above[:, 1], above[:, 0])
        p = np.column_stack([1 - above[:, 0], above[:, 0] - above[:, 1], above[:, 1]])
        return p / p.sum(axis=1, keepdims=True)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]
