import os, json, math
import numpy as np
from typing import Dict, List, Tuple

from sklearn.model_selection import KFold
from scipy.stats import beta
from scipy.signal import butter, filtfilt, detrend as sp_detrend
from aeon.distances import dtw_distance


def dtw_dist(X_CT, Y_CT, band_frac: float | None = 0.1) -> float:
    """
    Multivariate DTW via aeon. Inputs are (C, T) arrays.
    band_frac in [0,1] sets Sakoe–Chiba window; None disables windowing.
    Returns a scalar distance (float).
    """
    window = None
    if band_frac is not None:
        bf = float(band_frac)
        window = 0.0 if bf < 0.0 else (1.0 if bf > 1.0 else bf)
    # aeon expects (dims, time) == (C, T)
    return float(dtw_distance(X_CT, Y_CT, window=window))


def binomial_clopper_pearson(k, n, alpha=0.05):
    if n <= 0:
        return (np.nan, np.nan)
    lo = 0.0 if k == 0 else float(beta.ppf(alpha/2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha/2, k + 1, n - k))
    return lo, hi


def compute_eer(genuine_scores: np.ndarray, impostor_scores: np.ndarray) -> Tuple[float, float, float, float]:
    """
    Compute EER by sweeping τ over all unique score values.
    Assumes higher score = more anomalous; reject if score >= τ.
    Returns (eer, tau_eer, frr_at_eer, far_at_eer).
    """
    g = np.asarray(genuine_scores, dtype=float).ravel()
    i = np.asarray(impostor_scores, dtype=float).ravel()
    if g.size == 0 or i.size == 0:
        return (np.nan, np.nan, np.nan, np.nan)

    g.sort()
    i.sort()
    thresholds = np.unique(np.concatenate([g, i]))
    g_lt = np.searchsorted(g, thresholds, side="left")
    FRR = (g.size - g_lt) / g.size
    i_lt = np.searchsorted(i, thresholds, side="left")
    FAR = i_lt / i.size
    diffs = np.abs(FRR - FAR)
    j = int(np.argmin(diffs))
    tau_best = float(thresholds[j])
    frr_best = float(FRR[j])
    far_best = float(FAR[j])

    if 0 < j < thresholds.size - 1:
        j2 = j-1 if (FRR[j] - FAR[j]) * (FRR[j-1] - FAR[j-1]) <= 0 else j+1
        if abs(FRR[j2] - FAR[j2]) < abs(FRR[j] - FAR[j]):
            return float((FRR[j2] + FAR[j2]) / 2.0), float(thresholds[j2]), float(FRR[j2]), float(FAR[j2])
    eer = (frr_best + far_best) / 2.0
    return float(eer), tau_best, frr_best, far_best


def _load_seq(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    seq = [(d.get("timestampMillis"), d.get("values"))
           for d in data if "timestampMillis" in d and "values" in d]
    seq = [(t, v) for (t, v) in seq if isinstance(v, list) and len(v) > 0]
    if not seq: return None, None
    seq.sort(key=lambda x: x[0])
    times = [t for t,_ in seq]; vals = [v for _,v in seq]
    return times, vals


def _pad_or_trim(window, L):
    if window is None or not window: return None
    if len(window) < L: window = window + [window[-1]] * (L - len(window))
    elif len(window) > L: window = window[:L]
    return window


def _get_timestamps(sample_folder):
    touch_dir = os.path.join(sample_folder, "touch")
    if not os.path.isdir(touch_dir): return None, None
    jf = next((f for f in os.listdir(touch_dir) if f.endswith(".json")), None)
    if not jf: return None, None
    with open(os.path.join(touch_dir, jf), encoding="utf-8") as f: events = json.load(f)
    ts_start = next((e.get("timestamp") for e in events if e.get("variant") == "SELFIE_START"), None)
    ts_cap   = next((e.get("timestamp") for e in events if e.get("variant") == "SELFIE_CAPTURE"), None)
    return ts_start, ts_cap

def _zscore_CT(X: np.ndarray) -> np.ndarray:
    """
    Per-channel z-score along time for a single (C, T) array.
    Returns array with same shape.
    """
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True)
    return (X - mu) / (sd + 1e-8)


def _mag3(row):
    if not isinstance(row, (list, tuple)) or len(row) < 3: return None
    x, y, z = float(row[0]), float(row[1]), float(row[2])
    return float((x*x + y*y + z*z) ** 0.5)


def _apply_butterworth(arr, fs_hz, btype="low", cutoff_hz=20.0, order=2):
    """Apply a Butterworth filter to (C, T) array along time."""
    A = np.asarray(arr, dtype=np.float32)
    if A.ndim == 1:
        A = A[None, :]
    nyq = 0.5 * float(fs_hz)
    if isinstance(cutoff_hz, (list, tuple, np.ndarray)):
        cutoff = np.asarray(cutoff_hz, dtype=float).ravel()
        wn = np.clip(cutoff / nyq, 1e-6, 0.999)
        if btype == "band":
            if wn.size != 2:
                raise ValueError("For btype='band', cutoff_hz must be (low, high).")
            wn = np.sort(wn)
        else:
            wn = float(wn[0])
    else:
        wn = float(np.clip(float(cutoff_hz) / nyq, 1e-6, 0.999))
    b, a = butter(int(order), wn, btype=btype)
    return filtfilt(b, a, A, axis=-1).astype(np.float32)


def _remove_dc_1d(series, method: str = "mean"):
    x = np.asarray(series, dtype=np.float32).ravel()
    b = np.median(x) if method.lower() == "median" else np.mean(x)
    y = x - b
    return y.astype(float).tolist()


def _detrend_1d(series, mode: str = "linear"):
    x = np.asarray(series, dtype=np.float32).ravel()
    if mode not in {"linear", "constant"}:
        raise ValueError("MAG_DETREND_TYPE must be 'linear' or 'constant'")
    y = sp_detrend(x, type=mode)
    return y.astype(float).tolist()


def _unitdiff_T3(B: np.ndarray, demean=True, zscore=True) -> np.ndarray:
    """
    Unit-vector differencing on B (T,3)…
    Returns (T,3).
    """
    eps = 1e-8
    B = np.asarray(B, np.float32)
    if demean:
        B = B - B.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(B, axis=1, keepdims=True)
    u = B / (norms + eps)
    dU = np.diff(u, axis=0, prepend=u[:1])
    if zscore:
        mu = dU.mean(axis=0, keepdims=True)
        sd = dU.std(axis=0, ddof=1, keepdims=True)
        dU = (dU - mu) / (sd + eps)
    return dU.astype(np.float32)
