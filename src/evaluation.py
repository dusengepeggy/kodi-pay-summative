import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def evaluate(model, X, y):
    """Accuracy, macro-F1 and one-vs-rest ROC-AUC for a fitted model."""
    y = y.astype(str)
    pred = model.predict(X)
    out = {"accuracy": accuracy_score(y, pred),
           "macro_f1": f1_score(y, pred, average="macro")}
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        out["roc_auc_ovr"] = roc_auc_score(
            y, proba, multi_class="ovr", labels=model.classes_)
    return out


def accuracy_by_group(model, X, y, group):
    """Accuracy within each value of a grouping column (e.g. Year, IsYouth)."""
    y = y.astype(str)
    correct = pd.Series(model.predict(X) == y.values, index=X.index)
    return correct.groupby(group.values).mean().round(3)