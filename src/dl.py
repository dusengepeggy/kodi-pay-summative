from pathlib import Path

import joblib
import numpy as np
import tensorflow as tf

from src.evaluation import TIER_NUM, TIER_ORDER
from src.experiments import ExperimentTracker, set_seeds
from src.pipeline import build_category_indexer, build_preprocessor


def make_dataset(X, y=None, batch_size=256, shuffle=False, seed=42):
    """tf.data pipeline: slices -> (shuffle) -> batch -> prefetch.
    X is an array or a dict of arrays (one per model input)."""
    ds = tf.data.Dataset.from_tensor_slices(X if y is None else (X, y))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(y), seed=seed, reshuffle_each_iteration=True)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def _dense_block(x, hidden, dropout, l2):
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    for units in hidden:
        x = tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg)(x)
        x = tf.keras.layers.Dropout(dropout)(x)
    return x


def build_mlp(n_features, hidden=(256, 128), dropout=0.3, l2=0.0, lr=1e-3):
    """Sequential MLP on the preprocessed feature matrix."""
    reg = tf.keras.regularizers.l2(l2) if l2 else None
    layers = [tf.keras.Input((n_features,))]
    for units in hidden:
        layers += [tf.keras.layers.Dense(units, activation="relu", kernel_regularizer=reg),
                   tf.keras.layers.Dropout(dropout)]
    layers.append(tf.keras.layers.Dense(len(TIER_ORDER), activation="softmax"))
    model = tf.keras.Sequential(layers)
    model.compile(tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def build_functional(n_dense, cardinalities, hidden=(256, 128), dropout=0.3,
                     l2=0.0, lr=1e-3):
    """Functional model: a multi-hot/numeric branch plus one embedding per
    categorical column, concatenated before the final dense layers."""
    dense_in = tf.keras.Input((n_dense,), name="dense")
    branch = _dense_block(dense_in, hidden[:1], dropout, l2)
    inputs, embedded = [dense_in], [branch]
    for col, n in cardinalities.items():
        inp = tf.keras.Input((1,), dtype="int32", name=col)
        dim = min(8, (n + 1) // 2)  # small embedding for small vocabularies
        emb = tf.keras.layers.Embedding(n, dim)(inp)
        inputs.append(inp)
        embedded.append(tf.keras.layers.Flatten()(emb))
    x = tf.keras.layers.Concatenate()(embedded)
    x = _dense_block(x, hidden[1:], dropout, l2)
    out = tf.keras.layers.Dense(len(TIER_ORDER), activation="softmax")(x)
    model = tf.keras.Model(inputs, out)
    model.compile(tf.keras.optimizers.Adam(lr), loss="sparse_categorical_crossentropy",
                  metrics=["accuracy"])
    return model


class KerasPipeline:
    """Preprocessor + Keras model with a sklearn-like predict/predict_proba,
    so the same evaluation code works for both families."""

    def __init__(self, kind="mlp", cat_cols=(), pre_kwargs=None, hidden=(256, 128),
                 dropout=0.3, l2=0.0, lr=1e-3, batch_size=256, epochs=60,
                 patience=5, reduce_lr=False, seed=42):
        self.kind, self.cat_cols, self.pre_kwargs = kind, tuple(cat_cols), pre_kwargs or {}
        self.hidden, self.dropout, self.l2, self.lr = tuple(hidden), dropout, l2, lr
        self.batch_size, self.epochs, self.patience = batch_size, epochs, patience
        self.reduce_lr, self.seed = reduce_lr, seed

    def get_params(self, deep=True):
        return {k: v for k, v in vars(self).items() if not k.endswith("_")}

    def _inputs(self, X):
        dense = self.pre_.transform(X).astype(np.float32)
        if not self.cat_cols:
            return dense
        codes = self.cat_.transform(X[list(self.cat_cols)]).astype(np.int32) + 1  # 0 = unseen
        return {"dense": dense, **{c: codes[:, [i]] for i, c in enumerate(self.cat_cols)}}

    def fit(self, X, y, X_val, y_val, log_dir=None):
        set_seeds(self.seed)
        tf.config.experimental.enable_op_determinism()
        self.pre_ = build_preprocessor(**self.pre_kwargs).fit(X)
        if self.cat_cols:
            self.cat_ = build_category_indexer(list(self.cat_cols)).fit(X[list(self.cat_cols)])
            cards = {c: len(cats) + 1 for c, cats in
                     zip(self.cat_cols, self.cat_.named_steps["codes"].categories_)}
        Xt, Xv = self._inputs(X), self._inputs(X_val)
        yt = y.astype(str).map(TIER_NUM).to_numpy()
        yv = y_val.astype(str).map(TIER_NUM).to_numpy()
        n_dense = (Xt["dense"] if self.cat_cols else Xt).shape[1]

        if self.kind == "mlp":
            self.model_ = build_mlp(n_dense, self.hidden, self.dropout, self.l2, self.lr)
        else:
            self.model_ = build_functional(n_dense, cards, self.hidden, self.dropout,
                                           self.l2, self.lr)
        callbacks = [tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=self.patience, restore_best_weights=True)]
        if self.reduce_lr:
            callbacks.append(tf.keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5))
        if log_dir:
            callbacks.append(tf.keras.callbacks.TensorBoard(str(log_dir)))

        hist = self.model_.fit(
            make_dataset(Xt, yt, self.batch_size, shuffle=True, seed=self.seed),
            validation_data=make_dataset(Xv, yv, 2048),
            epochs=self.epochs, callbacks=callbacks, shuffle=False, verbose=0)
        self.history_ = {k: [float(v) for v in vals] for k, vals in hist.history.items()}
        self.classes_ = np.array(TIER_ORDER)
        return self

    def predict_proba(self, X):
        return self.model_.predict(make_dataset(self._inputs(X), batch_size=4096), verbose=0)

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(axis=1)]

    def save(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        self.model_.save(path / "model.keras")
        keras_model, self.model_ = self.model_, None  # Keras model is saved separately
        joblib.dump(self, path / "pipeline.joblib")
        self.model_ = keras_model

    @staticmethod
    def load(path):
        path = Path(path)
        obj = joblib.load(path / "pipeline.joblib")
        obj.model_ = tf.keras.models.load_model(path / "model.keras")
        return obj


class KerasTracker(ExperimentTracker):
    """Same fit-or-load and logging as ExperimentTracker, for KerasPipeline models.
    Saves models/exp_{id}/ (model.keras + pipeline.joblib) and the epoch history."""

    model_ext = ""
    log_group = "keras"

    def _save_model(self, model, exp_id):
        model.save(self.model_path(exp_id))

    def load(self, exp_id):
        return KerasPipeline.load(self.model_path(exp_id))

    def _fit(self, model, X, y):
        return model.fit(X, y, self.val, self.y_val,
                         log_dir=self.log_dir / self.log_group / self._run_id)

    def _extra_info(self, model):
        return {"epochs_trained": len(model.history_["loss"]), "history": model.history_}
