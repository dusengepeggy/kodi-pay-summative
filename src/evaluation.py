from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             f1_score, precision_recall_curve, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import StratifiedKFold, learning_curve

TIER_ORDER = ["Low", "Mid", "High"]
TIER_NUM = {t: i for i, t in enumerate(TIER_ORDER)}
FIG_DIR = Path(__file__).resolve().parents[1] / "figures"


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


def extra_metrics(model, X, y):
    """Ordinal-aware errors (how many tiers off) and youth-only scores."""
    y = y.astype(str).to_numpy()
    pred = np.asarray(model.predict(X)).astype(str)
    dist = np.abs(pd.Series(y).map(TIER_NUM).to_numpy()
                  - pd.Series(pred).map(TIER_NUM).to_numpy())
    youth = X["IsYouth"].to_numpy(dtype=bool)
    return {
        "mean_tier_error": dist.mean(),
        "far_error_rate": (dist == 2).mean(),
        "youth_accuracy": (pred[youth] == y[youth]).mean(),
        "youth_macro_f1": f1_score(y[youth], pred[youth], average="macro"),
    }


def all_metrics(model, X, y):
    """evaluate() plus extra_metrics() in one dict."""
    return {**evaluate(model, X, y), **extra_metrics(model, X, y)}


def accuracy_by_group(model, X, y, group):
    """Accuracy within each value of a grouping column (e.g. Year, IsYouth)."""
    y = y.astype(str)
    correct = pd.Series(model.predict(X) == y.values, index=X.index)
    return correct.groupby(group.values).mean().round(3)


def subgroup_table(model, X, y, group):
    """Per-group size, accuracy, macro-F1 and the group's own majority baseline."""
    df = pd.DataFrame({"y": y.astype(str).to_numpy(),
                       "pred": np.asarray(model.predict(X)).astype(str),
                       "g": np.asarray(group)})
    rows = []
    for g, d in df.groupby("g"):
        rows.append({"group": g, "n": len(d),
                     "accuracy": accuracy_score(d.y, d.pred),
                     "macro_f1": f1_score(d.y, d.pred, average="macro"),
                     "majority_class": d.y.mode().iat[0],
                     "majority_baseline_acc": (d.y == d.y.mode().iat[0]).mean()})
    return pd.DataFrame(rows).set_index("group").round(3)


def _save(fig, name):
    if name:
        FIG_DIR.mkdir(exist_ok=True)
        fig.savefig(FIG_DIR / name, dpi=200, bbox_inches="tight")


def plot_confusions(y, pred, masks, title_prefix="", name=None):
    """Row-normalized confusion matrices, one panel per {title: boolean mask}."""
    y, pred = np.asarray(y).astype(str), np.asarray(pred).astype(str)
    fig, axes = plt.subplots(1, len(masks), figsize=(5.5 * len(masks), 4.5))
    for ax, (title, m) in zip(np.atleast_1d(axes), masks.items()):
        ConfusionMatrixDisplay.from_predictions(
            y[m], pred[m], labels=TIER_ORDER, normalize="true",
            cmap="Blues", ax=ax, values_format=".2f", colorbar=False)
        ax.set_title(f"{title_prefix}{title} (n={m.sum():,})")
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_roc_pr(model, X, y, title="", name=None):
    """One-vs-rest ROC and precision-recall curves per tier."""
    y = np.asarray(y).astype(str)
    proba = model.predict_proba(X)
    classes = list(model.classes_)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for tier in TIER_ORDER:
        p = proba[:, classes.index(tier)]
        fpr, tpr, _ = roc_curve(y == tier, p)
        ax1.plot(fpr, tpr, label=f"{tier} (AUC={auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(y == tier, p)
        ax2.plot(rec, prec, label=f"{tier} (AUC={auc(rec, prec):.3f})")
    ax1.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax2.axhline(1 / 3, color="k", ls="--", lw=0.8, label="chance (1/3)")
    ax1.set(xlabel="False positive rate", ylabel="True positive rate",
            title=f"ROC one-vs-rest {title}")
    ax2.set(xlabel="Recall", ylabel="Precision", title=f"Precision-recall {title}")
    ax1.legend(); ax2.legend()
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_learning_curve(estimator, X, y, title="", name=None, seed=42,
                        sizes=(0.1, 0.25, 0.5, 0.75, 1.0), n_jobs=-1):
    """sklearn learning curve (macro-F1, 3-fold CV on the data given)."""
    cv = StratifiedKFold(3, shuffle=True, random_state=seed)
    n, tr, va = learning_curve(estimator, X, y.astype(str), cv=cv,
                               train_sizes=list(sizes), scoring="f1_macro",
                               n_jobs=n_jobs, shuffle=True, random_state=seed)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    for scores, label in [(tr, "train"), (va, "cross-validation")]:
        m, s = scores.mean(1), scores.std(1)
        ax.plot(n, m, "o-", label=label)
        ax.fill_between(n, m - s, m + s, alpha=0.2)
    ax.set(xlabel="Training rows", ylabel="Macro-F1", title=f"Learning curve {title}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, name)
    table = pd.DataFrame({"train_rows": n, "train_f1": tr.mean(1),
                          "cv_f1": va.mean(1), "gap": tr.mean(1) - va.mean(1)})
    return fig, table.round(3)


def plot_history(history, title="", name=None):
    """Keras loss and accuracy per epoch (train vs validation)."""
    h = pd.DataFrame(history)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    h[["loss", "val_loss"]].plot(ax=ax1, title=f"Loss {title}")
    h[["accuracy", "val_accuracy"]].plot(ax=ax2, title=f"Accuracy {title}")
    for ax in (ax1, ax2):
        ax.set_xlabel("Epoch"); ax.grid(alpha=0.3)
    fig.tight_layout()
    _save(fig, name)
    return fig
