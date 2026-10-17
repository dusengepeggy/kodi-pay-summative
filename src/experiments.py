import hashlib
import json
import os
import platform
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from src.evaluation import all_metrics

# NumPy and SciPy each ship an OpenBLAS with 16 threads that fight each other here;
# one BLAS thread made LogisticRegression ~50x faster (190 s -> 4 s).
threadpool_limits(1, user_api="blas")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
# Files whose code changes what a model sees; a change here forces a refit
CODE_FILES = ["preprocessing.py", "features.py", "splits.py",
              "transformers.py", "pipeline.py"]


def set_seeds(seed=42):
    """Seed Python, NumPy, the hash seed and (if installed) TensorFlow."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
    except ImportError:
        pass


def _hash(obj):
    text = json.dumps(obj, sort_keys=True, default=str)
    return hashlib.sha256(text.encode()).hexdigest()[:12]


def data_hash(frames, target):
    """Hash of the row ids and labels, so a changed split or label is detected."""
    h = hashlib.sha256()
    for f in frames:
        key = f[["Year", "ResponseId", target]].astype(str)
        h.update(pd.util.hash_pandas_object(key, index=False).values.tobytes())
    return h.hexdigest()[:12]


def code_hash(files=CODE_FILES):
    h = hashlib.sha256()
    for name in files:
        h.update((SRC_DIR / name).read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()[:12]


def model_params(model):
    """All hyperparameters as text; memory addresses removed so the hash is stable."""
    return {k: re.sub(r" at 0x[0-9a-fA-F]+", "", repr(v))
            for k, v in sorted(model.get_params(deep=True).items())}


def library_versions():
    import sklearn
    out = {"python": platform.python_version(), "numpy": np.__version__,
           "pandas": pd.__version__, "scikit-learn": sklearn.__version__}
    if "tensorflow" in sys.modules:
        out["tensorflow"] = sys.modules["tensorflow"].__version__
    return out


class ExperimentTracker:
    """Fit-or-load sklearn experiments and log each one to disk.

    Per experiment: models/exp_{id}.joblib, results/settings/exp_{id}.json,
    one row in results/experiments.csv and TensorBoard scalars in logs/.
    """

    model_ext = ".joblib"
    log_group = "sklearn"

    def __init__(self, train, val, target="PayTier", seed=42, root=PROJECT_ROOT):
        self.train, self.val, self.target, self.seed = train, val, target, seed
        self.y_train = train[target].astype(str)
        self.y_val = val[target].astype(str)
        self.root = Path(root)
        self.models_dir = self.root / "models"
        self.settings_dir = self.root / "results" / "settings"
        self.csv_path = self.root / "results" / "experiments.csv"
        self.log_dir = self.root / "logs"
        for d in (self.models_dir, self.settings_dir):
            d.mkdir(parents=True, exist_ok=True)

    # ---- paths and persistence -------------------------------------------
    def model_path(self, exp_id):
        return self.models_dir / f"exp_{exp_id}{self.model_ext}"

    def json_path(self, exp_id):
        return self.settings_dir / f"exp_{exp_id}.json"

    def _save_model(self, model, exp_id):
        joblib.dump(model, self.model_path(exp_id), compress=3)

    def load(self, exp_id):
        return joblib.load(self.model_path(exp_id))

    def _fit(self, model, X, y):
        model.fit(X, y)
        return model

    def _config(self, name, model, settings):
        return {"name": name, "settings": settings, "seed": self.seed,
                "params": model_params(model)}

    # ---- main entry point --------------------------------------------------
    def run(self, exp_id, name, model, settings, note="", force=False,
            train=None):
        """Fit (or reload) `model`, score it on train and val, and log it.
        `train` overrides the training rows (used by the population experiment)."""
        set_seeds(self.seed)
        train = self.train if train is None else train
        y_train = train[self.target].astype(str)

        hashes = {"config_hash": _hash(self._config(name, model, settings)),
                  "data_hash": data_hash([train, self.val], self.target),
                  "code_hash": code_hash()}
        saved = self._read_json(exp_id)
        same = saved is not None and all(saved.get(k) == v for k, v in hashes.items())

        if same and not force and self.model_path(exp_id).exists():
            print(f"[exp {exp_id}] hashes match -> loading saved model, no refit")
            model, fit_seconds = self.load(exp_id), saved["fit_seconds"]
        else:
            if saved is not None and not force:
                changed = [k for k, v in hashes.items() if saved.get(k) != v]
                print(f"[exp {exp_id}] refitting, changed: {changed or ['model file missing']}")
            t0 = time.time()
            model = self._fit(model, train, y_train)
            fit_seconds = round(time.time() - t0, 1)
            self._save_model(model, exp_id)
            print(f"[exp {exp_id}] fitted in {fit_seconds}s")

        metrics = {f"train_{k}": v for k, v in all_metrics(model, train, y_train).items()}
        metrics.update({f"val_{k}": v for k, v in all_metrics(model, self.val, self.y_val).items()})

        if force and saved is not None:
            diffs = {k: abs(metrics[k] - saved["metrics"][k])
                     for k in metrics if k in saved["metrics"]}
            worst = max(diffs, key=diffs.get)
            print(f"[exp {exp_id}] largest metric difference vs saved run: "
                  f"{diffs[worst]:.2e} ({worst})")

        self._record(exp_id, name, model, settings, note, hashes, metrics,
                     fit_seconds, len(train))
        return model

    # ---- logging -------------------------------------------------------------
    def _read_json(self, exp_id):
        p = self.json_path(exp_id)
        return json.loads(p.read_text()) if p.exists() else None

    def _extra_info(self, model):
        """Search results worth keeping (best params and CV score)."""
        if hasattr(model, "best_params_"):
            return {"best_params": model.best_params_,
                    "cv_best_score": float(model.best_score_)}
        return {}

    def _record(self, exp_id, name, model, settings, note, hashes, metrics,
                fit_seconds, n_train):
        run_id = f"exp{exp_id}-{hashes['config_hash'][:8]}"
        info = {"exp_id": exp_id, "name": name, "run_id": run_id,
                "settings": settings, "note": note, "seed": self.seed,
                "params": model_params(model) if hasattr(model, "get_params") else {},
                **hashes, "versions": library_versions(),
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "fit_seconds": fit_seconds, "n_train": n_train,
                "n_val": len(self.val), "metrics": metrics,
                **self._extra_info(model)}
        self.json_path(exp_id).write_text(json.dumps(info, indent=2, default=str))

        row = {"id": exp_id, "run_id": run_id, "model": name,
               "settings": json.dumps(settings, default=str),
               "best_params": json.dumps(info.get("best_params", ""), default=str),
               "n_train": n_train, "n_val": len(self.val),
               **{k: round(v, 4) for k, v in metrics.items()},
               "fit_seconds": fit_seconds, "note": note}
        table = pd.read_csv(self.csv_path) if self.csv_path.exists() else pd.DataFrame()
        if len(table):
            table = table[table["id"].astype(str) != str(exp_id)]
        table = pd.concat([table, pd.DataFrame([row])], ignore_index=True)
        table.to_csv(self.csv_path, index=False)
        self._tensorboard(run_id, metrics, settings)

    def _tensorboard(self, run_id, metrics, settings):
        try:
            import tensorflow as tf
        except ImportError:
            return
        writer = tf.summary.create_file_writer(str(self.log_dir / self.log_group / run_id))
        with writer.as_default():
            for k, v in metrics.items():
                tf.summary.scalar(k.replace("_", "/", 1), v, step=0)
            tf.summary.text("settings", json.dumps(settings, default=str), step=0)
        writer.flush()

    def table(self):
        """The experiment table, one row per experiment, sorted by id."""
        t = pd.read_csv(self.csv_path)
        return t.sort_values("id", key=lambda s: s.astype(str).str.zfill(4)).reset_index(drop=True)
