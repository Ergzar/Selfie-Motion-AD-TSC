# Detector implementations + factory (imported by the benchmark)
# - Plain one-class: IsolationForest, One-Class SVM, LSTM-AE
# - QUANT featurizer variants: IF, OCSVM, kNN (+ optional power transform)
# - ROCKAD wrapper (optional; only if installed)
# - Inline detectors moved from main: knn_series, iforest_raw, knn_dtw
# - Factory: make_detector(name, cfg)

from __future__ import annotations
import numpy as np

# sklearn
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import PowerTransformer, StandardScaler

# Helpers used by inline detectors
from ad_utils import _zscore_CT, dtw_dist

# QUANT transformer (aeon)
try:
    from aeon.transformations.collection.interval_based import QUANTTransformer
    _HAS_QUANT = True
except Exception:
    QUANTTransformer = None
    _HAS_QUANT = False


# Base API to access detector implementations

class BaseDetector:
    """Minimal interface: fit(X_list), score(X_list) -> higher means 'more anomalous'."""
    def fit(self, X_list): raise NotImplementedError
    def score(self, X_list): raise NotImplementedError


# Plain detectors 

class IForestDetector(BaseDetector):
    def __init__(self, n_estimators=300, max_samples="auto", contamination=0.1, random_state=7):
        self.clf = IsolationForest(
            n_estimators=n_estimators,
            max_samples=max_samples,
            contamination=contamination,
            random_state=random_state,
            n_jobs=-1,
        )
    def _flatten(self, X_list):
        X = np.asarray(X_list, dtype=np.float32)   # (N, C, T)
        return X.reshape((X.shape[0], -1))
    def fit(self, X_list):
        self.clf.fit(self._flatten(X_list))
        return self
    def score(self, X_list):
        s = self.clf.score_samples(self._flatten(X_list))  # higher = more normal
        return -np.asarray(s, dtype=float)                 # invert: higher = more anomalous


class OCSVMDetector(BaseDetector):
    def __init__(self, nu=0.05, gamma="scale"):
        self.clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    def _flatten(self, X_list):
        X = np.asarray(X_list, dtype=np.float32)
        return X.reshape((X.shape[0], -1))
    def fit(self, X_list):
        self.clf.fit(self._flatten(X_list))
        return self
    def score(self, X_list):
        d = self.clf.decision_function(self._flatten(X_list))  # higher = more normal
        return -np.asarray(d, dtype=float)


class RockadDetector(BaseDetector):
    """Wrapper for ROCKAD (if available)."""
    def __init__(self, random_state=7, **kw):
        try:
            from whole_ROCKAD import wROCKAD
        except Exception as e:
            raise RuntimeError("ROCKAD not available. Install ROCKADs.") from e
        self._clf = wROCKAD(random_state=random_state, normalise=False, **kw)

    def fit(self, X_list):
        X = np.asarray(X_list, dtype=np.float32)
        self._clf.fit(X)
        return self

    def score(self, X_list):
        X = np.asarray(X_list, dtype=np.float32)
        return np.asarray(self._clf.predict(X), dtype=float)


# QUANT + tabular one-class 

class _BaseQuantTabular(BaseDetector):
    """QUANT featurizer with optional power transform and scaler."""
    def __init__(self, interval_depth=6, quantile_divisor=4,
                 power_transform=False, power_method="yeo-johnson",
                 power_standardize=False, scale_after_power=False):
        if not _HAS_QUANT:
            raise RuntimeError("QUANTTransformer not available. Install aeon>=0.8.")
        self.quant = QUANTTransformer(
            interval_depth=interval_depth, quantile_divisor=quantile_divisor
        )
        self._pt = PowerTransformer(method=power_method, standardize=power_standardize) \
                   if power_transform else None
        self._sc = StandardScaler() if scale_after_power else None

    def _fit_features(self, X_list):
        X = np.asarray(X_list, dtype=np.float32)
        F = self.quant.fit_transform(X)
        if self._pt is not None: F = self._pt.fit_transform(F)
        if self._sc is not None: F = self._sc.fit_transform(F)
        return F

    def _to_features(self, X_list):
        X = np.asarray(X_list, dtype=np.float32)
        F = self.quant.transform(X)
        if self._pt is not None: F = self._pt.transform(F)
        if self._sc is not None: F = self._sc.transform(F)
        return F


class QuantIForestDetector(_BaseQuantTabular):
    def __init__(self, n_estimators=300, max_samples="auto", contamination=0.1,
                 interval_depth=6, quantile_divisor=4,
                 power_transform=False, power_method="yeo-johnson",
                 power_standardize=False, scale_after_power=False, random_state=7):
        super().__init__(interval_depth, quantile_divisor,
                         power_transform, power_method,
                         power_standardize, scale_after_power)
        self.clf = IsolationForest(
            n_estimators=n_estimators, max_samples=max_samples,
            contamination=contamination, random_state=random_state, n_jobs=-1
        )
    def fit(self, X_list):
        F = self._fit_features(X_list)
        self.clf.fit(F)
        return self
    def score(self, X_list):
        F = self._to_features(X_list)
        s = self.clf.score_samples(F)  # higher normal
        return -np.asarray(s, dtype=float)


class QuantOCSVMDetector(_BaseQuantTabular):
    def __init__(self, nu=0.05, gamma="scale",
                 interval_depth=6, quantile_divisor=4,
                 power_transform=False, power_method="yeo-johnson",
                 power_standardize=False, scale_after_power=False):
        super().__init__(interval_depth, quantile_divisor,
                         power_transform, power_method,
                         power_standardize, scale_after_power)
        self.clf = OneClassSVM(kernel="rbf", nu=nu, gamma=gamma)
    def fit(self, X_list):
        F = self._fit_features(X_list)
        self.clf.fit(F)
        return self
    def score(self, X_list):
        F = self._to_features(X_list)
        d = self.clf.decision_function(F)  # higher normal
        return -np.asarray(d, dtype=float)


class QuantKNNDetector(_BaseQuantTabular):
    """k-NN distance on QUANT features (mean or k-th neighbor distance)."""
    def __init__(self, k=1, metric="euclidean", agg="mean",
                 interval_depth=6, quantile_divisor=4,
                 power_transform=False, power_method="yeo-johnson",
                 power_standardize=False, scale_after_power=False):
        super().__init__(interval_depth, quantile_divisor,
                         power_transform, power_method,
                         power_standardize, scale_after_power)
        self.k = max(1, int(k))
        self.metric = metric
        self.agg = str(agg).lower()
        self._nn = None
        self._F_train = None

    def fit(self, X_list):
        F = self._fit_features(X_list)
        n_neighbors = min(self.k, max(1, F.shape[0]))
        self._nn = NearestNeighbors(n_neighbors=n_neighbors, metric=self.metric)
        self._nn.fit(F)
        self._F_train = F
        return self

    def score(self, X_list):
        if self._nn is None or self._F_train is None:
            raise RuntimeError("QuantKNNDetector not fitted.")
        F = self._to_features(X_list)
        d, _ = self._nn.kneighbors(F, n_neighbors=min(self.k, self._F_train.shape[0]),
                                   return_distance=True)
        if self.agg == "kth":
            s = d[:, -1]
        else:
            s = d.mean(axis=1)
        return s.astype(float)


# Inline detectors (moved from main) 

class _BaseOneClassKNN:
    def __init__(self, k:int, agg:str):
        self.k = int(max(1, k))
        self.agg = agg.lower()
        self._train: np.ndarray | None = None  # (N, ...)

    def fit(self, X: np.ndarray):
        self._train = np.asarray(X, dtype=np.float32)
        return self

    def _aggregate(self, dists: np.ndarray) -> float:
        if dists.size == 0:
            return float("nan")
        k = min(self.k, dists.size)
        idx = np.argpartition(dists, k - 1)[:k]
        topk = dists[idx]
        if self.agg == "kth":
            return float(np.max(topk))
        return float(np.mean(topk))

    def score(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class _KNNSeries(_BaseOneClassKNN):
    """Euclidean on raw (C,T) (optionally per-channel z-score)."""
    def __init__(self, k:int, agg:str, zscore:bool):
        super().__init__(k, agg)
        self.zscore = bool(zscore)
        self._train_flat: np.ndarray | None = None  # (N, C*T)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        super().fit(X)
        Xp = self._train
        if self.zscore:
            Xp = np.stack([_zscore_CT(x) for x in Xp], axis=0)
        N, C, T = Xp.shape
        self._train_flat = Xp.reshape(N, C*T).astype(np.float32, copy=False)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._train_flat is None:
            raise RuntimeError("Detector not fit.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, ...]
        if self.zscore:
            Xp = np.stack([_zscore_CT(x) for x in X], axis=0)
        else:
            Xp = X
        Xf = Xp.reshape(Xp.shape[0], -1).astype(np.float32, copy=False)
        scores = np.empty(Xf.shape[0], dtype=np.float32)
        for i in range(Xf.shape[0]):
            diffs = self._train_flat - Xf[i]
            dists = np.sqrt(np.sum(diffs * diffs, axis=1, dtype=np.float64)).astype(np.float64)
            scores[i] = self._aggregate(dists)
        return scores


class _IForestRaw:
    """
    IsolationForest on raw series:
      - Optional per-channel z-score along time (shape-only) before flattening
      - Flatten (C,T) → (C*T) features
      - Score = -decision_function(...) so that higher = more anomalous
    """
    def __init__(self, n_estimators, max_samples, contamination, zscore: bool, random_state: int):
        from sklearn.ensemble import IsolationForest as _SkIsolationForest 
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self.zscore = bool(zscore)
        self.random_state = int(random_state)
        self._iforest = None  # set in fit

    def _prep(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, ...]
        if self.zscore:
            X = np.stack([_zscore_CT(x) for x in X], axis=0)
        N, C, T = X.shape
        return X.reshape(N, C*T).astype(np.float32, copy=False)

    def fit(self, X: np.ndarray):
        Xf = self._prep(X)
        self._iforest = IsolationForest(
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            contamination=self.contamination,
            random_state=self.random_state,
            n_jobs=-1,
        ).fit(Xf)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._iforest is None:
            raise RuntimeError("Detector not fit.")
        Xf = self._prep(X)
        s = self._iforest.decision_function(Xf).astype(np.float32)  # higher=more normal
        return (-s)  # higher = more anomalous


class _KNNDtw(_BaseOneClassKNN):
    """
    k-NN with DTW distance on raw multivariate series:
      - optional per-channel z-score over time (shape-only)
      - distance = multivariate DTW (L2 at each align step)
      - higher score = more anomalous (aggregate of k nearest distances)
    """
    def __init__(self, k:int, agg:str, zscore:bool, band_frac: float | None):
        super().__init__(k, agg)
        self.zscore = bool(zscore)
        self.band_frac = band_frac
        self._train_seq: np.ndarray | None = None  # (N, C, T)

    def fit(self, X: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 3:
            raise ValueError("Expected X with shape (N, C, T)")
        if self.zscore:
            X = np.stack([_zscore_CT(x) for x in X], axis=0)
        self._train_seq = X
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        if self._train_seq is None:
            raise RuntimeError("Detector not fit.")
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 2:
            X = X[None, ...]
        if self.zscore:
            X = np.stack([_zscore_CT(x) for x in X], axis=0)

        Tr = self._train_seq  # shape (Ntr, C, T)
        Ntr = Tr.shape[0]

        scores = np.empty(X.shape[0], dtype=np.float32)
        for i in range(X.shape[0]):
            Xi_CT = X[i]                # (C, T)
            Ti = Xi_CT.shape[1]
            dists = np.empty(Ntr, dtype=np.float64)
            for j in range(Ntr):
                Yj_CT = Tr[j]           # (C, T)
                Tj = Yj_CT.shape[1]
                d = dtw_dist(Xi_CT, Yj_CT, band_frac=self.band_frac)
                dists[j] = float(d) / (Ti + Tj)   # length norm
            scores[i] = self._aggregate(dists)
        return scores.astype(np.float32)


# Factory & helpers

def quant_available() -> bool:
    return _HAS_QUANT

def detector_names() -> list[str]:
    base = ["rockad", "iforest", "ocsvm", "lstm_ae",
            "knn_series", "iforest_raw", "knn_dtw"]
    quant = ["quant_iforest", "quant_ocsvm", "quant_knn"] if _HAS_QUANT else []
    return base + quant

def make_detector(name: str, cfg: dict) -> BaseDetector:
    """
    name: one of detector_names()
    cfg: {
      "random_seed": int,
      "rockad": {...},
      "iforest": {"n_estimators","max_samples","contamination"},
      "ocsvm": {"nu","gamma"},
      "lstm_ae": {"latent","epochs","batch","val_split","patience","l2","verbose"},
      "quant": {"interval_depth","quantile_divisor","power_transform","power_method",
                "power_standardize","scale_after_power"},
      "knn": {"k","metric","agg"},
      "knn_series": {"k","agg","zscore"},
      "IFOREST_ZSCORE": bool,
      "knn_dtw_k": int, "knn_dtw_agg": str, "knn_dtw_zscore": bool, "knn_dtw_band_frac": float|None
    }
    """
    n = name.lower()
    rs = cfg.get("random_seed", 7)

    if n == "rockad":
        params = dict(cfg.get("rockad", {}))
        return RockadDetector(random_state=rs, **params)

    if n == "iforest":
        p = cfg.get("iforest", {})
        return IForestDetector(
            n_estimators=p.get("n_estimators", 300),
            max_samples=p.get("max_samples", "auto"),
            contamination=p.get("contamination", 0.1),
            random_state=rs,
        )

    if n == "ocsvm":
        p = cfg.get("ocsvm", {})
        return OCSVMDetector(
            nu=p.get("nu", 0.05),
            gamma=p.get("gamma", "scale"),
        )

    if n == "lstm_ae":
        p = cfg.get("lstm_ae", {})
        return LSTMAEDetector(
            latent=p.get("latent", 32),
            epochs=p.get("epochs", 30),
            batch=p.get("batch", 32),
            val_split=p.get("val_split", 0.1),
            patience=p.get("patience", 5),
            l2=p.get("l2", 1e-4),
            verbose=p.get("verbose", 0),
        )

    # QUANT wrappers
    q = cfg.get("quant", {})
    if n == "quant_iforest":
        p = cfg.get("iforest", {})
        return QuantIForestDetector(
            n_estimators=p.get("n_estimators", 300),
            max_samples=p.get("max_samples", "auto"),
            contamination=p.get("contamination", 0.1),
            interval_depth=q.get("interval_depth", 6),
            quantile_divisor=q.get("quantile_divisor", 4),
            power_transform=q.get("power_transform", False),
            power_method=q.get("power_method", "yeo-johnson"),
            power_standardize=q.get("power_standardize", False),
            scale_after_power=q.get("scale_after_power", False),
            random_state=rs,
        )

    if n == "quant_ocsvm":
        p = cfg.get("ocsvm", {})
        return QuantOCSVMDetector(
            nu=p.get("nu", 0.05),
            gamma=p.get("gamma", "scale"),
            interval_depth=q.get("interval_depth", 6),
            quantile_divisor=q.get("quantile_divisor", 4),
            power_transform=q.get("power_transform", False),
            power_method=q.get("power_method", "yeo-johnson"),
            power_standardize=q.get("power_standardize", False),
            scale_after_power=q.get("scale_after_power", False),
        )

    if n == "quant_knn":
        kn = cfg.get("knn", {})
        return QuantKNNDetector(
            k=kn.get("k", 1),
            metric=kn.get("metric", "euclidean"),
            agg=kn.get("agg", "mean"),
            interval_depth=q.get("interval_depth", 6),
            quantile_divisor=q.get("quantile_divisor", 4),
            power_transform=q.get("power_transform", False),
            power_method=q.get("power_method", "yeo-johnson"),
            power_standardize=q.get("power_standardize", False),
            scale_after_power=q.get("scale_after_power", False),
        )

    # Inline detectors from main 

    if n == "knn_series":
        c = cfg.get("knn_series", {})
        return _KNNSeries(k=c.get("k", 1), agg=c.get("agg", "kth"), zscore=c.get("zscore", True))

    if n == "iforest_raw":
        p = cfg.get("iforest", {})
        return _IForestRaw(
            n_estimators=p.get("n_estimators", 300),
            max_samples=p.get("max_samples", "auto"),
            contamination=p.get("contamination", "auto"),
            zscore=cfg.get("IFOREST_ZSCORE", False),
            random_state=rs,
        )

    if n == "knn_dtw":
        return _KNNDtw(
            k=cfg.get("knn_dtw_k", 3),
            agg=cfg.get("knn_dtw_agg", "kth"),
            zscore=cfg.get("knn_dtw_zscore", True),
            band_frac=cfg.get("knn_dtw_band_frac", 0.1),
        )

    raise ValueError(f"Unknown detector name: {name}")


# LSTM Autoencoder (working version)

class LSTMAEDetector(BaseDetector):
    """
    LSTM sequence autoencoder (reconstruction MSE as anomaly score).
    Safer training:
      - No Masking() (avoids accidental masking of near-zero z-scored data)
      - Manual validation split so EarlyStopping always has val_loss
      - Cached tf.function for fast inference
    """
    def __init__(self, latent=32, epochs=50, batch=32, val_split=0.1, patience=5, l2=1e-4, verbose=1):
        self.latent = latent
        self.epochs = epochs
        self.batch = batch
        self.val_split = float(val_split)
        self.patience = patience
        self.l2 = l2
        self.verbose = verbose
        self.model = None
        self._infer = None

    def _to_seq(self, X_list):
        X = np.asarray(X_list, dtype=np.float32)  # (N,C,T)
        return np.transpose(X, (0, 2, 1))         # (N,T,C)

    def fit(self, X_list):
        try:
            import tensorflow as tf
            from tensorflow.keras import layers, regularizers, callbacks, Model
        except Exception as e:
            raise RuntimeError(
                "TensorFlow is required for LSTM autoencoder. Install tensorflow>=2."
            ) from e

        X = self._to_seq(X_list)
        N, nT, nC = X.shape

        # --- build ---
        inp = layers.Input(shape=(nT, nC))
        x   = layers.LSTM(self.latent * 2, return_sequences=True,
                          kernel_regularizer=regularizers.l2(self.l2))(inp)
        x   = layers.LSTM(self.latent, return_sequences=False,
                          kernel_regularizer=regularizers.l2(self.l2))(x)
        z   = layers.Dense(self.latent, activation="linear")(x)
        y   = layers.RepeatVector(nT)(z)
        y   = layers.LSTM(self.latent, return_sequences=True,
                          kernel_regularizer=regularizers.l2(self.l2))(y)
        out = layers.TimeDistributed(layers.Dense(nC))(y)

        model = Model(inp, out)
        model.compile(optimizer="adam", loss="mse")

        # Foolproof validation split
        cb = []
        use_val = False
        if self.val_split > 0.0 and N >= 2:
            n_val = max(1, int(np.floor(N * self.val_split)))
            n_val = min(n_val, N - 1)  # keep at least 1 train sample
            # last n_val as validation (shuffle=True is fine, but manual split is explicit)
            X_tr, X_val = X[:-n_val], X[-n_val:]
            cb.append(callbacks.EarlyStopping(
                monitor="val_loss", patience=self.patience, restore_best_weights=True
            ))
            use_val = True
        else:
            # fall back to train-only early stop 
            cb.append(callbacks.EarlyStopping(
                monitor="loss", patience=self.patience, restore_best_weights=True
            ))
            X_tr = X
            X_val = None

        if use_val:
            model.fit(
                X_tr, X_tr,
                validation_data=(X_val, X_val),
                epochs=self.epochs,
                batch_size=self.batch,
                shuffle=True,
                verbose=self.verbose,
                callbacks=cb,
            )
        else:
            model.fit(
                X_tr, X_tr,
                epochs=self.epochs,
                batch_size=self.batch,
                shuffle=True,
                verbose=self.verbose,
                callbacks=cb,
            )

        self.model = model

        # Cache a tf.function for faster inference
        try:
            spec = tf.TensorSpec(shape=(None, nT, nC), dtype=tf.float32)
            self._infer = tf.function(self.model, input_signature=[spec], reduce_retracing=True)
        except Exception:
            self._infer = None

        return self

    def score(self, X_list):
        X = self._to_seq(X_list).astype(np.float32)
        if self._infer is not None:
            import tensorflow as tf
            X_hat = self._infer(tf.convert_to_tensor(X)).numpy()
        else:
            X_hat = self.model.predict(X, batch_size=self.batch, verbose=0)
        err = np.mean((X_hat - X) ** 2, axis=(1, 2))
        return np.asarray(err, dtype=float)

