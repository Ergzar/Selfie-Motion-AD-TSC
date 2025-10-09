import os
import json
import numpy as np
from time import perf_counter

from sklearn.model_selection import GroupShuffleSplit, GroupKFold
from sklearn.metrics import roc_auc_score
from scipy.signal import butter, filtfilt
from sklearn.neighbors import NearestNeighbors

# Common AD helpers reused from utils.py 
from ad_utils import (
    _load_seq,
    _pad_or_trim,
    _get_timestamps,
    _mag3,
    _apply_butterworth,
    _unitdiff_T3,
    _zscore_CT,   # used by local raw knn
)

from detectors import make_detector as _mk_from_library

# SETTINGS
BASE_PATH                = os.path.expanduser("~/Desktop/Dataset+anomaly")
ANOMALOUS_CLASSES        = ["handheld", "stationary", "delayed_1.5-3"]  # spoof folders 

SAMPLES_BEFORE           = 50
SAMPLES_AFTER            = 100
TOTAL_LEN_RAW            = SAMPLES_BEFORE + SAMPLES_AFTER

MODE                     = "capture_only"  # {"capture_only","start_only","capture_plus_from_start"}
CAPTURE_REQUIRED         = True

# Concatenate N samples after START in front of captoure window (per-channel, along time)
CONCAT_START_IN_FRONT        = True
CONCAT_SAMPLES_AFTER_START   = 10

# Per-sensor axis mapping (ints 0/1/2, or "mag" for magnitude)
SENSORS = {
    "acc_":   [0,1,2],
    "gyro_":  [0,1,2],
    # "magnet_":[0],
    # "lin_acc_": ["mag"],
}

# Downsample
DOWNSAMPLE_200_TO_50HZ   = False
RAW_FS_HZ                = 50.0

# Butterworth (applied post-downsample)
APPLY_BUTTERWORTH        = True
BW_ORDER                 = 2
BW_BTYPE                 = "low"       # "low", "high", "band"
BW_CUTOFF_HZ             = 12.5        # float or (low,high) for band

# Magnetometer special processing 
MAG_PROC_MODE            = "unitdiff"  # "off" | "unitdiff"
MAG_UNITDIFF_DEMEAN      = True     # If false, context leak will persist
MAG_UNITDIFF_ZSCORE      = True     # Practically redundant (makes no difference)

DETECTOR                 = "lstm_ae"    #  rockad | iforest | ocsvm | lstm_ae | quant_iforest | knn_raw | quant_knn | all

# ROCKAD params
ROCKAD_PARAMS = dict(n_estimators=24, n_kernels=1024, n_neighbors=3, power_transform=True, n_jobs=-1)

# IsolationForest params
IFOREST_TREES            = 1500
IFOREST_MAX_SAMPLES      = "auto"
IFOREST_CONTAM           = "auto"

# One-Class SVM params
OCSVM_NU                 = 0.05
OCSVM_GAMMA              = "scale"

# LSTM Autoencoder params
LSTM_EPOCHS              = 50
LSTM_BATCH               = 32
LSTM_LATENT              = 24
LSTM_VERBOSE             = 1
LSTM_VAL_SPLIT           = 0.20
LSTM_PATIENCE            = 6
LSTM_L2                  = 5e-4

# QUANT feature params (used by quant_* detectors)
QUANT_INTERVAL_DEPTH     = 6
QUANT_QDIV               = 4

# --- k-NN detector params ---
KNN_K                    = 3          # neighbors
KNN_AGG                  = "mean"     # "mean" or "kth". 
KNN_METRIC               = "euclidean"
RAW_ZSCORE_FOR_KNN       = True       # per-channel z-score over time before flattening

# Train/test splits & calibration
NORMAL_TEST_RATIO        = 0.20
FRR_TARGET               = 0.01

# Repeated inner OOF calibration
MAX_INNER_FOLDS          = 8
INNER_REPEATS            = 5
OOF_AVERAGE_PER_SAMPLE   = True

# outer resamples for mean±std reporting 
OUTER_RESAMPLES          = 5

# Misc
RANDOM_SEED              = 7
IGNORE_FOLDERS           = []

def _window_capture_centered(file_path, ts_cap, ts_start):
    times, vals = _load_seq(file_path)
    if times is None: return None
    if ts_cap is None:
        if CAPTURE_REQUIRED: return None
        anchor = ts_start if ts_start is not None else times[0]
    else:
        anchor = ts_cap
    idx = int(np.clip(np.searchsorted(times, anchor), 0, len(times)))
    start = max(0, idx - SAMPLES_BEFORE)
    end   = min(len(times), idx + SAMPLES_AFTER)
    return _pad_or_trim(vals[start:end], TOTAL_LEN_RAW)

def _window_from_start(file_path):
    times, vals = _load_seq(file_path)
    if times is None: return None
    start_idx = 0
    end_idx   = min(len(times), start_idx + TOTAL_LEN_RAW)
    return _pad_or_trim(vals[start_idx:end_idx], TOTAL_LEN_RAW)

def _window_from_start_len(file_path, L, ts_start):
    times, vals = _load_seq(file_path)
    if times is None: return None
    if ts_start is not None:
        start_idx = int(np.clip(np.searchsorted(times, ts_start), 0, len(times)))
    else:
        start_idx = 0
    end_idx = min(len(times), start_idx + int(L))
    return _pad_or_trim(vals[start_idx:end_idx], int(L))

def _decimate_to_50hz(arr, order=4):
    if arr.ndim == 1: arr = arr[None, :]
    nyq = 0.5 * RAW_FS_HZ
    b, a = butter(order, 25.0/nyq, btype="low")
    filt = filtfilt(b, a, arr, axis=-1)
    return filt[..., ::4].astype(np.float32)


# Multichannel extraction (duplicate from user-vs-spoof for all intents and purposes, but uses too many constants to move to utils)

def extract_multichannel(sample_path, ts_start, ts_cap):
    use_capture = MODE in {"capture_only","capture_plus_from_start"} or CONCAT_START_IN_FRONT
    use_start   = MODE in {"start_only","capture_plus_from_start"}
    chans = []

    for prefix, axes in SENSORS.items():
        if not axes: continue
        file_path = None
        for root, _, files in os.walk(sample_path):
            for fn in files:
                if fn.startswith(prefix) and fn.endswith(".json"):
                    file_path = os.path.join(root, fn); break
            if file_path: break
        if not file_path: return None

        is_mag = prefix.lower().startswith("magnet")

        if CONCAT_START_IN_FRONT:
            w_front = _window_from_start_len(file_path, CONCAT_SAMPLES_AFTER_START, ts_start)
            w_cap   = _window_capture_centered(file_path, ts_cap, ts_start)
            if w_front is None or w_cap is None: return None
            width = len(w_cap[0])
            if any(len(x) != len(w_front[0]) for x in w_front) or any(len(x) != width for x in w_cap): return None

            if is_mag and MAG_PROC_MODE == "unitdiff":
                B_front = np.array([[row[0], row[1], row[2]] for row in w_front], dtype=np.float32)
                B_cap   = np.array([[row[0], row[1], row[2]] for row in w_cap],   dtype=np.float32)
                B = np.vstack([B_front, B_cap])
                dU = _unitdiff_T3(B, demean=MAG_UNITDIFF_DEMEAN, zscore=MAG_UNITDIFF_ZSCORE)
                for ax in axes:
                    if isinstance(ax, int) and ax in (0,1,2): chans.append(dU[:, int(ax)].astype(float).tolist())
                    elif isinstance(ax, str) and ax.lower() in ("mag","magnitude","norm","l2"): chans.append(np.linalg.norm(dU, axis=1).astype(float).tolist())
                    else: raise ValueError("SENSORS['magnet_'] bad axis")
                continue

            for ax in axes:
                if isinstance(ax, str) and ax.lower() in ("mag","magnitude","norm","l2"):
                    front_m = [_mag3(row) for row in w_front]; cap_m = [_mag3(row) for row in w_cap]
                    if any(m is None for m in front_m + cap_m): return None
                    series = front_m + cap_m
                else:
                    if not isinstance(ax, int) or ax < 0 or ax >= width: return None
                    front_ax = [row[ax] for row in w_front]; cap_ax = [row[ax] for row in w_cap]
                    series   = front_ax + cap_ax
                chans.append(series)
            continue

        # Non-concat
        built = []
        if use_capture:
            w_cap = _window_capture_centered(file_path, ts_cap, ts_start)
            if w_cap is None: return None
            built.append(w_cap)
        if use_start:
            w_start = _window_from_start(file_path)
            if w_start is None: return None
            built.append(w_start)
        if not built or any(len(w) != TOTAL_LEN_RAW for w in built): return None
        width = len(built[0][0])

        if is_mag and MAG_PROC_MODE == "unitdiff":
            for w in built:
                B = np.array([[row[0], row[1], row[2]] for row in w], dtype=np.float32)
                dU = _unitdiff_T3(B, demean=MAG_UNITDIFF_DEMEAN, zscore=MAG_UNITDIFF_ZSCORE)
                for ax in axes:
                    if isinstance(ax, int) and ax in (0,1,2): chans.append(dU[:, int(ax)].astype(float).tolist())
                    elif isinstance(ax, str) and ax.lower() in ("mag","magnitude","norm","l2"): chans.append(np.linalg.norm(dU, axis=1).astype(float).tolist())
                    else: raise ValueError("SENSORS['magnet_'] bad axis")
            continue

        for ax in axes:
            if isinstance(ax, str) and ax.lower() in ("mag","magnitude","norm","l2"):
                for w in built:
                    mags = []
                    for row in w:
                        m = _mag3(row)
                        if m is None: return None
                        mags.append(m)
                    chans.append(mags)
            else:
                if not isinstance(ax, int) or ax < 0 or ax >= width: return None
                for w in built: chans.append([row[ax] for row in w])

    arr = np.asarray(chans, dtype=np.float32)  # (C, T_raw)
    expected_raw_len = (CONCAT_SAMPLES_AFTER_START + TOTAL_LEN_RAW) if CONCAT_START_IN_FRONT else TOTAL_LEN_RAW
    if arr.shape[1] != expected_raw_len: return None

    if DOWNSAMPLE_200_TO_50HZ:
        arr = _decimate_to_50hz(arr); fs_effective = RAW_FS_HZ / 4.0
    else:
        fs_effective = RAW_FS_HZ

    if APPLY_BUTTERWORTH:
        arr = _apply_butterworth(arr, fs_effective, btype=BW_BTYPE, cutoff_hz=BW_CUTOFF_HZ, order=BW_ORDER)

    return arr

def load_dataset(base_path):
    X_norm, y_users = [], []
    spoofs_by = {t: [] for t in ANOMALOUS_CLASSES}
    for user in os.listdir(base_path):
        if user in IGNORE_FOLDERS: continue
        udir = os.path.join(base_path, user)
        if not os.path.isdir(udir): continue
        for sample in os.listdir(udir):
            sdir = os.path.join(udir, sample)
            if not os.path.isdir(sdir): continue
            ts_start, ts_cap = _get_timestamps(sdir)
            arr = extract_multichannel(sdir, ts_start, ts_cap)
            if arr is None: continue
            if user in ANOMALOUS_CLASSES:
                spoofs_by[user].append(arr)
            else:
                X_norm.append(arr); y_users.append(user)
    return X_norm, y_users, spoofs_by

class KNNRawDetector:
    def __init__(self, k=5, agg="mean", metric="euclidean", zscore=True):
        self.k = int(max(1,k)); self.agg = agg; self.metric = metric; self.zscore = zscore
        self.nn = None
    def _to_features(self, X_list):
        X = np.asarray(X_list, dtype=np.float32)  # shape is (N,C,T)
        if self.zscore:
            X = np.stack([_zscore_CT(x) for x in X], axis=0)  # reuse utils._zscore_CT
        return X.reshape((X.shape[0], -1))        # flatten
    def fit(self, X_list):
        F = self._to_features(X_list)
        self.nn = NearestNeighbors(n_neighbors=self.k, metric=self.metric, n_jobs=-1)
        self.nn.fit(F); return self
    def score(self, X_list):
        F = self._to_features(X_list)
        dists, _ = self.nn.kneighbors(F, n_neighbors=self.k, return_distance=True)
        if self.agg == "kth":
            s = dists[:, self.k-1]
        else:
            s = dists.mean(axis=1)
        return np.asarray(s, dtype=float)  # higher = more anomalous

def _detector_cfg() -> dict:
    """Map this script's constants to detectors.make_detector(...) config."""
    return {
        "random_seed": RANDOM_SEED,
        "rockad": { **ROCKAD_PARAMS },
        "iforest": {
            "n_estimators": IFOREST_TREES,
            "max_samples": IFOREST_MAX_SAMPLES,
            "contamination": IFOREST_CONTAM,
        },
        "ocsvm": {
            "nu": OCSVM_NU,
            "gamma": OCSVM_GAMMA,
        },
        "lstm_ae": {
            "latent": LSTM_LATENT,
            "epochs": LSTM_EPOCHS,
            "batch": LSTM_BATCH,
            "val_split": LSTM_VAL_SPLIT,
            "patience": LSTM_PATIENCE,
            "l2": LSTM_L2,
            "verbose": LSTM_VERBOSE,
        },
        "quant": {
            "interval_depth": QUANT_INTERVAL_DEPTH,
            "quantile_divisor": QUANT_QDIV,
        },
        # 'knn' block unused here, as raw-kNN gets handled locally
    }

def make_detector(name: str):
    n = name.lower()
    if n == "knn_raw":
        return KNNRawDetector(k=KNN_K, agg=KNN_AGG, metric=KNN_METRIC, zscore=RAW_ZSCORE_FOR_KNN)
    # Delegate all others to detectors.py
    return _mk_from_library(n, _detector_cfg())


def _randomized_group_labels(groups, rng):
    g = np.array(groups); uniq = np.unique(g)
    pref = rng.randint(0, 10**9, size=len(uniq))
    mapping = {u: f"{int(p):09d}_{i}" for i, (u, p) in enumerate(zip(uniq, pref))}
    return np.array([mapping[u] for u in g])

def grouped_fit_test_split(X, groups, test_ratio, seed):
    if len(X) < 3: return X, [], [], []
    gss = GroupShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
    idx = np.arange(len(X))
    (fit_idx, test_idx) = next(gss.split(idx, groups=groups))
    X_fit  = [X[i] for i in fit_idx]; g_fit  = [groups[i] for i in fit_idx]
    X_test = [X[i] for i in test_idx]; g_test = [groups[i] for i in test_idx]
    return X_fit, X_test, g_fit, g_test

# ---------------- One-resample runner (single detector) ----------------

def run_one_resample(detector_name, seed_offset=0):
    # Load dataset
    X_normals, y_users, spoofs_by = load_dataset(BASE_PATH)
    users = sorted(set(y_users))
    num_spoofs_total = sum(len(v) for v in spoofs_by.values())

    if num_spoofs_total == 0:
        raise SystemExit("No spoofs found. Check ANOMALOUS_CLASSES and data paths.")

    # Grouped fit/test split on normals
    X_fit_norm, X_test_norm, g_fit, g_test = grouped_fit_test_split(
        X_normals, y_users, NORMAL_TEST_RATIO, RANDOM_SEED + seed_offset
    )
    if not X_test_norm:
        raise SystemExit("Grouped split produced empty normal test set; adjust NORMAL_TEST_RATIO.")

    # Repeated OOF calibration (τ at FRR_TARGET on normals)
    unique_groups = np.unique(g_fit)
    max_folds = int(max(2, min(MAX_INNER_FOLDS, unique_groups.size)))
    if max_folds < 2:
        det_tmp = make_detector(detector_name)
        det_tmp.fit(X_fit_norm)
        scores_oof = det_tmp.score(X_fit_norm).tolist()
    else:
        fit_n = len(X_fit_norm)
        oof_sum = np.zeros(fit_n, dtype=np.float64); oof_cnt = np.zeros(fit_n, dtype=np.int32)
        rng = np.random.RandomState(RANDOM_SEED + 123 + seed_offset)
        base_idx = np.arange(fit_n)
        for _ in range(int(max(1, INNER_REPEATS))):
            rand_groups = _randomized_group_labels(np.array(g_fit), rng)
            gkf = GroupKFold(n_splits=max_folds)
            for tr_idx, cal_idx in gkf.split(base_idx, groups=rand_groups):
                det_fold = make_detector(detector_name)
                det_fold.fit([X_fit_norm[i] for i in tr_idx])
                scores_fold = det_fold.score([X_fit_norm[i] for i in cal_idx])
                oof_sum[cal_idx] += scores_fold.astype(float); oof_cnt[cal_idx] += 1
        valid = oof_cnt > 0
        if not np.any(valid):
            raise SystemExit("Repeated OOF produced no calibration scores.")
        if OOF_AVERAGE_PER_SAMPLE:
            scores_oof = (oof_sum[valid] / oof_cnt[valid]).tolist()
        else:
            scores_oof = []
            for i in range(fit_n):
                if oof_cnt[i] > 0:
                    mean_i = oof_sum[i] / oof_cnt[i]
                    scores_oof.extend([float(mean_i)] * int(oof_cnt[i]))

    tau_train = float(np.quantile(np.asarray(scores_oof, dtype=float), 1.0 - FRR_TARGET))

    # Train final detector on ALL fit normals
    detector = make_detector(detector_name)
    detector.fit(X_fit_norm)

    # Scores
    scores_norm_test = detector.score(X_test_norm)  # normals = 0
    spoof_scores_type = {t: (detector.score(spoofs_by[t]) if len(spoofs_by[t]) else np.array([]))
                         for t in ANOMALOUS_CLASSES}

    # Timing (fairly redundant)
    all_val = X_test_norm.copy()
    for t in ANOMALOUS_CLASSES:
        all_val.extend(spoofs_by[t])
    t0 = perf_counter()
    _ = detector.score(all_val) if len(all_val) > 0 else np.array([])
    t1 = perf_counter()
    ms_per_sample = ( (t1 - t0) / max(1, len(all_val)) ) * 1000.0

    # Metrics @ τ(train)
    FRR = float((scores_norm_test >= tau_train).mean()) if scores_norm_test.size else np.nan
    FAR_by_type = {}
    total_spoofs = 0
    total_acc_spoofs = 0
    for t, sc in spoof_scores_type.items():
        if sc.size:
            FAR_by_type[t] = float((sc < tau_train).mean())
            total_spoofs += sc.size
            total_acc_spoofs += int((sc < tau_train).sum())
        else:
            FAR_by_type[t] = np.nan

    # Pooled FAR across all spoof types (counts-weighted)
    pooled_FAR = (total_acc_spoofs / total_spoofs) if total_spoofs > 0 else np.nan

    # AUROC (unused in thesis)
    try:
        all_spoof_scores = np.concatenate([v for v in spoof_scores_type.values() if v.size > 0], axis=0)
    except ValueError:
        all_spoof_scores = np.array([], dtype=float)
    if scores_norm_test.size > 0 and all_spoof_scores.size > 0:
        y_true = np.concatenate([np.zeros_like(scores_norm_test, dtype=int),
                                 np.ones_like(all_spoof_scores, dtype=int)])
        y_score = np.concatenate([scores_norm_test, all_spoof_scores]).astype(float)
        AUROC = float(roc_auc_score(y_true, y_score))  # higher score = more anomalous
    else:
        AUROC = np.nan

    return FRR, FAR_by_type, pooled_FAR, AUROC, ms_per_sample

# Multi-resample + (optionally) multi-detector 

def run_detector(detector_name):
    FRRs = []
    FARs_by_type_list = []
    pooled_FARs = []
    AUROCs = []
    msps = []

    for r in range(OUTER_RESAMPLES):
        FRR, FAR_by_type, pooled_FAR, auroc, msp = run_one_resample(detector_name, seed_offset=r)
        FRRs.append(FRR)
        FARs_by_type_list.append(FAR_by_type)
        pooled_FARs.append(pooled_FAR)
        AUROCs.append(auroc)
        msps.append(msp)

    def mstd(a):
        a = np.asarray(a, dtype=float)
        if a.size == 0: return np.nan, np.nan
        if a.size == 1: return float(a[0]), np.nan
        return float(np.mean(a)), float(np.std(a, ddof=1))

    # Aggregate per-type FARs
    type_to_vals = {t: [] for t in ANOMALOUS_CLASSES}
    for d in FARs_by_type_list:
        for t in ANOMALOUS_CLASSES:
            v = d.get(t, np.nan)
            if not np.isnan(v): type_to_vals[t].append(v)

    res = {
        "detector": detector_name,
        "FRR_m": mstd([v for v in FRRs if not np.isnan(v)])[0],
        "FRR_s": mstd([v for v in FRRs if not np.isnan(v)])[1],
        "FAR_by_type": {t: mstd(type_to_vals[t]) for t in ANOMALOUS_CLASSES},
        "pooledFAR_m": mstd([v for v in pooled_FARs if not np.isnan(v)])[0],
        "pooledFAR_s": mstd([v for v in pooled_FARs if not np.isnan(v)])[1],
        "AUROC_m": mstd([v for v in AUROCs if not np.isnan(v)])[0],
        "AUROC_s": mstd([v for v in AUROCs if not np.isnan(v)])[1],
        "msps_m": mstd(msps)[0],
        "msps_s": mstd(msps)[1],
    }
    return res

def main():
    np.random.seed(RANDOM_SEED)

    if DETECTOR.lower() == "all":
        to_run = ["rockad","iforest","ocsvm","lstm_ae","quant_iforest","knn_raw","quant_knn"]
    else:
        to_run = [DETECTOR]

    results = [run_detector(name) for name in to_run]

    # Print fancy summary 
    sep = "="*66
    print("\n" + sep)
    print("RESULTS @ τ(train)  +  AUROC (threshold-independent)")
    print(sep)

    for res in results:
        name = res["detector"]
        FRR_m, FRR_s = res["FRR_m"], res["FRR_s"]
        AUROC_m, AUROC_s = res["AUROC_m"], res["AUROC_s"]
        msps_m, msps_s = res["msps_m"], res["msps_s"]
        print(f"\n[{name}]")
        if np.isnan(FRR_s) or OUTER_RESAMPLES == 1:
            print(f"  FRR (normals): {FRR_m:.4f}")
            print(f"  AUROC        : {AUROC_m:.4f}")
            print(f"  ms/sample    : {msps_m:.2f}")
        else:
            print(f"  FRR (normals): {FRR_m:.4f} ± {FRR_s:.4f}")
            print(f"  AUROC        : {AUROC_m:.4f} ± {AUROC_s:.4f}")
            print(f"  ms/sample    : {msps_m:.2f} ± {msps_s:.2f}")

        for t in ANOMALOUS_CLASSES:
            m, s = res["FAR_by_type"][t]
            fr_m, fr_s = res["FRR_m"], res["FRR_s"]
            if np.isnan(m):
                print(f"  {t:>12} → FAR: (no data)    | FRR: {fr_m:.4f}" if (np.isnan(fr_s) or OUTER_RESAMPLES == 1)
                      else f"  {t:>12} → FAR: (no data)    | FRR: {fr_m:.4f} ± {fr_s:.4f}")
            else:
                if np.isnan(s) or OUTER_RESAMPLES == 1:
                    print(f"  {t:>12} → FAR: {m:.4f}         | FRR: {fr_m:.4f}")
                else:
                    print(f"  {t:>12} → FAR: {m:.4f} ± {s:.4f} | FRR: {fr_m:.4f} ± {fr_s:.4f}")

        p_m, p_s = res["pooledFAR_m"], res["pooledFAR_s"]
        if np.isnan(p_s) or OUTER_RESAMPLES == 1:
            print(f"  Pooled FAR (all spoofs): {p_m:.4f}")
        else:
            print(f"  Pooled FAR (all spoofs): {p_m:.4f} ± {p_s:.4f}")

    #  Compact table for run-all 
    if len(results) > 1:
        print("\n" + "-"*110)
        hdr = (
            f"{'Detector':<16} | {'FRR':>10} | "
            + " | ".join([f"FAR[{t}]" for t in ANOMALOUS_CLASSES])
            + f" | {'Pooled FAR':>10} | {'AUROC':>8} | {'ms/sample':>10}"
        )
        print(hdr)
        print("-"*110)
        def fmt(m,s):
            return f"{m:>.4f}" if np.isnan(s) or OUTER_RESAMPLES==1 else f"{m:.4f}±{s:.4f}"
        for r in results:
            row = [
                f"{r['detector']:<16}",
                f"{fmt(r['FRR_m'], r['FRR_s']):>10}",
            ]
            for t in ANOMALOUS_CLASSES:
                m,s = r["FAR_by_type"][t]
                row.append(f"{fmt(m,s):>10}")
            row.append(f"{fmt(r['pooledFAR_m'], r['pooledFAR_s']):>10}")
            row.append(f"{fmt(r['AUROC_m'], r['AUROC_s']):>8}")
            row.append(f"{fmt(r['msps_m'], r['msps_s']):>10}")
            print(" | ".join(row))
        print("-"*110)

if __name__ == "__main__":
    main()
