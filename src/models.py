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


class SavedModelEnsemble(BaseEstimator, ClassifierMixin):
    """Soft voting over already-saved experiments: the tier probabilities of each
    member are averaged (fixed weights, nothing is refitted). Only the experiment
    ids are pickled; the member models are reloaded from models/ when needed."""

    def __init__(self, members=(), weights=None):
        self.members = members  # e.g. (("sklearn", "6g"), ("keras", "9d"))
        self.weights = weights

    def _load(self):
        from pathlib import Path
        import joblib
        root = Path(__file__).resolve().parents[1] / "models"
        models = []
        for kind, exp_id in self.members:
            if kind == "keras":
                from src.dl import KerasPipeline
                models.append(KerasPipeline.load(root / f"exp_{exp_id}"))
            else:
                models.append(joblib.load(root / f"exp_{exp_id}.joblib"))
        self.models_ = models

    def fit(self, X, y=None):
        self._load()
        self.classes_ = np.array(TIER_ORDER)
        return self

    def predict_proba(self, X):
        if getattr(self, "models_", None) is None:
            self._load()
        w = np.ones(len(self.members)) if self.weights is None else np.asarray(self.weights, float)
        total = 0
        for weight, m in zip(w, self.models_):
            p = m.predict_proba(X)
            order = [list(m.classes_).index(t) for t in TIER_ORDER]  # align columns to Low/Mid/High
            total = total + weight * p[:, order]
        return total / w.sum()

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def __getstate__(self):
        state = self.__dict__.copy()
        state.pop("models_", None)  # members are saved as their own experiments
        return state
