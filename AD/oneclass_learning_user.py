from __future__ import annotations
import os, json, math
import numpy as np
from typing import Dict, List, Tuple

from sklearn.model_selection import KFold

from detectors import make_detector
from ad_utils import ( 
    binomial_clopper_pearson,
    compute_eer,
    _load_seq, _pad_or_trim, _get_timestamps, _mag3,
    _apply_butterworth, _unitdiff_T3, _remove_dc_1d, _detrend_1d
)

BASE_PATH               = os.path.expanduser("~/Desktop/Dataset")
IGNORE_FOLDERS          = []
RANDOM_SEED             = 7

# Which detectors to run (in order). Leave empty to run the single DETECTOR below.
RUN_LIST = []

# Detector choice if RUN_LIST == []
#  "rockad"|"iforest"|"ocsvm"|"quant_iforest"|"quant_knn"|"knn_series"|"iforest_raw"|"knn_dtw"|"lstm_ae"
DETECTOR                = "knn_Dtw"

# Time window (around SELFIE_CAPTURE or START fallback)
SAMPLES_BEFORE          = 50
SAMPLES_AFTER           = 150
TOTAL_LEN               = SAMPLES_BEFORE + SAMPLES_AFTER

# Windowing mode
MODE                    = "capture_only"  # {"capture_only","start_only","capture_plus_from_start"}
CAPTURE_REQUIRED        = True

# Concatenate N samples from after SELFIE_START in front of capture window
CONCAT_START_IN_FRONT       = True
CONCAT_SAMPLES_AFTER_START  = 10

# Sensors to load. ints = axis index; "mag" = sqrt(x^2+y^2+z^2)
SENSORS = {
    "acc_":  [0],
    "gyro_": [0],
    "magnet_": [0],   
    # "lin_acc_": ["mag"]
}

# ROCKAD params 
ROCKAD_PARAMS = dict(n_estimators=14, n_kernels=512, n_neighbors=3, power_transform=True, n_jobs=-1)

# IForest params
IFOREST_TREES           = 300
IFOREST_MAX_SAMPLES     = "auto"
IFOREST_CONTAM          = "auto"
IFOREST_ZSCORE          = True  # if True and DETECTOR == "iforest", we'll route to "iforest_raw"

# OCSVM params
OCSVM_NU                = 0.05
OCSVM_GAMMA             = "scale"

# QUANT featurizer params
QUANT_INTERVAL_DEPTH    = 6
QUANT_QDIV              = 4

# QUANT kNN detector settings
KNN_K                   = 1
KNN_METRIC              = "euclidean"
KNN_AGG                 = "kth"          # {"mean","kth"}
KNN_QUANT_ZSCORE        = False          # z-score QUANT features per feature using train mean/std

# Inline-equivalent: Raw-series kNN (Euclidean)
KNN_SERIES_K            = 3
KNN_SERIES_AGG          = "kth"          # {"mean","kth"}
KNN_SERIES_ZSCORE       = True           # per-channel z-score over time (shape-only)

# Inline-equivalent: DTW→kNN settings
KNN_DTW_K               = 3
KNN_DTW_AGG             = "kth"
KNN_DTW_ZSCORE          = True
KNN_DTW_BAND_FRAC       = 0.1            # Sakoe–Chiba band width fraction in [0,1]; None disables

# LSTM Autoencoder params
LSTM_EPOCHS             = 100
LSTM_BATCH              = 32
LSTM_LATENT             = 32
LSTM_VERBOSE            = 0
LSTM_VAL_SPLIT          = 0.10
LSTM_PATIENCE           = 5
LSTM_L2                 = 1e-4

# Protocol
USER_MIN_SAMPLES        = 12
FRR_TARGET              = 0.01           # τ at (1 - FRR_TARGET) quantile of OOF genuine
OOF_FOLDS_MAX           = 2              # inner OOF folds (2 means 5 training 5 calibration)
OOF_REPEATS             = 5              # number of OOF resamples
TRAIN_MIN_SAMPLES       = 10             # per-user, per-fold exact train size

# Global Butterworth on (C,T) (12.5 Hz & 2nd order used in thesis)
APPLY_BUTTERWORTH       = True
BW_ORDER                = 2
BW_BTYPE                = "low"          
BW_CUTOFF_HZ            = 12.5
FILTER_FS_HZ            = 50.0

#   "off" | "dc" | "detrend" | "unitdiff" (unitdiff always used in thesis)
MAG_PROC_MODE           = "unitdiff"

# DC removal (mag-only)
MAG_DC_METHOD           = "mean"         # {"mean","median"}

# Detrend (mag-only)
MAG_DETREND_TYPE        = "constant"     # {"linear","constant"}

# Unit-vector differencing (mag-only)
MAG_UNITDIFF_DEMEAN     = True          
MAG_UNITDIFF_ZSCORE     = False         

# Threshold safety margin  to further reduce genuine rejections
#   "mult" or "off", multiplies by (1 + margin_value)
TAU_MARGIN_MODE   = "off"
TAU_MARGIN_VALUE  = 0.1
TAU_SIGMA_MULT    = 1.0
TAU_Q_MAX         = 0.999

# Dataset window extraction utils

def _window_from_start_len(file_path, L, ts_start):
    times, vals = _load_seq(file_path)
    if times is None: return None
    if ts_start is not None:
        start_idx = int(np.clip(np.searchsorted(times, ts_start), 0, len(times)))
    else:
        start_idx = 0
    end_idx = min(len(times), start_idx + int(L))
    return _pad_or_trim(vals[start_idx:end_idx], int(L))

def _window_from_start(file_path):
    return _window_from_start_len(file_path, TOTAL_LEN, None)

def _window_capture_centered(file_path, ts_cap, ts_start):
    times, vals = _load_seq(file_path)
    if times is None: return None
    anchor = ts_cap if ts_cap is not None else (ts_start if not CAPTURE_REQUIRED and ts_start is not None else times[0])
    if anchor is None and CAPTURE_REQUIRED:
        return None
    idx = int(np.clip(np.searchsorted(times, anchor), 0, len(times)))
    start = max(0, idx - SAMPLES_BEFORE)
    end   = min(len(times), idx + SAMPLES_AFTER)
    return _pad_or_trim(vals[start:end], TOTAL_LEN)

def extract_multichannel(sample_path, ts_start, ts_cap):
    """
    Build (C, T) for one sample folder.
    """
    use_capture = MODE in {"capture_only","capture_plus_from_start"} or CONCAT_START_IN_FRONT
    use_start   = MODE in {"start_only","capture_plus_from_start"}

    chans = []
    for prefix, axes in SENSORS.items():
        if not axes:
            continue

        file_path = None
        for root, _, files in os.walk(sample_path):
            for fn in files:
                if fn.startswith(prefix) and fn.endswith(".json"):
                    file_path = os.path.join(root, fn); break
            if file_path: break
        if not file_path:
            return None

        is_mag = prefix.lower().startswith("magnet")

        # ----- CONCAT path -----
        if CONCAT_START_IN_FRONT:
            w_front = _window_from_start_len(file_path, CONCAT_SAMPLES_AFTER_START, ts_start)
            w_cap   = _window_capture_centered(file_path, ts_cap, ts_start)
            if w_front is None or w_cap is None:
                return None
            width = len(w_cap[0])
            if any(len(x) != len(w_front[0]) for x in w_front) or any(len(x) != width for x in w_cap):
                return None

            if is_mag and MAG_PROC_MODE == "unitdiff":
                B_front = np.array([[row[0], row[1], row[2]] for row in w_front], dtype=np.float32)
                B_cap   = np.array([[row[0], row[1], row[2]] for row in w_cap],   dtype=np.float32)
                B = np.vstack([B_front, B_cap])
                dU = _unitdiff_T3(B, demean=MAG_UNITDIFF_DEMEAN, zscore=MAG_UNITDIFF_ZSCORE)

                want_mag = any(isinstance(ax, str) and ax.lower() in ("mag","magnitude","norm","l2") for ax in axes)
                req_axes = [ax for ax in axes if isinstance(ax, int) and 0 <= ax <= 2]

                if want_mag:
                    chans.append(np.linalg.norm(dU, axis=1).astype(float).tolist())
                for ax in req_axes:
                    chans.append(dU[:, ax].astype(float).tolist())

            else:
                for ax in axes:
                    if isinstance(ax, str) and ax.lower() in ("mag","magnitude","norm","l2"):
                        front_m = [_mag3(row) for row in w_front]
                        cap_m   = [_mag3(row) for row in w_cap]
                        if any(m is None for m in front_m + cap_m): return None
                        series = front_m + cap_m
                    else:
                        if not isinstance(ax, int) or ax < 0 or ax >= width: return None
                        front_ax = [row[ax] for row in w_front]
                        cap_ax   = [row[ax] for row in w_cap]
                        series   = front_ax + cap_ax

                    if is_mag:
                        if MAG_PROC_MODE == "dc":
                            series = _remove_dc_1d(series, method=MAG_DC_METHOD)
                        elif MAG_PROC_MODE == "detrend":
                            series = _detrend_1d(series, mode=MAG_DETREND_TYPE)
                    chans.append(series)

        # NON-CONCAT path 

        else:
            built = []
            if use_capture:
                w_cap = _window_capture_centered(file_path, ts_cap, ts_start)
                if w_cap is None: return None
                built.append(w_cap)
            if use_start:
                w_start = _window_from_start(file_path)
                if w_start is None: return None
                built.append(w_start)

            if not built or any(len(w) != TOTAL_LEN for w in built): return None
            width = len(built[0][0])

            if is_mag and MAG_PROC_MODE == "unitdiff":
                want_mag = any(isinstance(ax, str) and ax.lower() in ("mag","magnitude","norm","l2") for ax in axes)
                req_axes = [ax for ax in axes if isinstance(ax, int) and 0 <= ax <= 2]

                for w in built:
                    B = np.array([[row[0], row[1], row[2]] for row in w], dtype=np.float32)
                    dU = _unitdiff_T3(B, demean=MAG_UNITDIFF_DEMEAN, zscore=MAG_UNITDIFF_ZSCORE)

                    if want_mag:
                        chans.append(np.linalg.norm(dU, axis=1).astype(float).tolist())
                    for ax in req_axes:
                        chans.append(dU[:, ax].astype(float).tolist())

            else:
                for w in built:
                    for ax in axes:
                        if isinstance(ax, str) and ax.lower() in ("mag","magnitude","norm","l2"):
                            series = []
                            for row in w:
                                m = _mag3(row)
                                if m is None: return None
                                series.append(m)
                        else:
                            if not isinstance(ax, int) or ax < 0 or ax >= width: return None
                            series = [row[ax] for row in w]

                        if is_mag:
                            if MAG_PROC_MODE == "dc":
                                series = _remove_dc_1d(series, method=MAG_DC_METHOD)
                            elif MAG_PROC_MODE == "detrend":
                                series = _detrend_1d(series, mode=MAG_DETREND_TYPE)
                        chans.append(series)

    arr = np.asarray(chans, dtype=np.float32)  # (C, T)
    expected_len = (CONCAT_SAMPLES_AFTER_START + TOTAL_LEN) if CONCAT_START_IN_FRONT else TOTAL_LEN
    if arr.shape[1] != expected_len:
        return None

    if APPLY_BUTTERWORTH:
        arr = _apply_butterworth(arr, fs_hz=FILTER_FS_HZ, btype=BW_BTYPE,
                                 cutoff_hz=BW_CUTOFF_HZ, order=BW_ORDER)
    return arr

def load_normals_by_user(base_path) -> Dict[str, np.ndarray]:
    """Returns dict: user -> array (N_u, C, T) of genuine samples."""
    by_user: Dict[str, List[np.ndarray]] = {}
    for user in os.listdir(base_path):
        if user in IGNORE_FOLDERS:
            continue
        udir = os.path.join(base_path, user)
        if not os.path.isdir(udir):
            continue
        for sample in os.listdir(udir):
            sdir = os.path.join(udir, sample)
            if not os.path.isdir(sdir):
                continue
            ts_start, ts_cap = _get_timestamps(sdir)
            arr = extract_multichannel(sdir, ts_start, ts_cap)
            if arr is None:
                continue
            by_user.setdefault(user, []).append(arr)
    for u in list(by_user.keys()):
        if not by_user[u]:
            del by_user[u]
        else:
            by_user[u] = np.asarray(by_user[u], dtype=np.float32)
    return by_user

# Detector factory 
def _detector_cfg() -> dict:
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
        # QUANT featurizer config
        "quant": {
            "interval_depth": QUANT_INTERVAL_DEPTH,
            "quantile_divisor": QUANT_QDIV,
        },
        # QUANT detector params
        "knn": {
            "k": KNN_K,
            "metric": KNN_METRIC,
            "agg": KNN_AGG,
            "zscore": KNN_QUANT_ZSCORE,
        },
        # Raw-series kNN params
        "knn_series": {
            "k": KNN_SERIES_K, "agg": KNN_SERIES_AGG, "zscore": KNN_SERIES_ZSCORE,
        },
        # knn_dtw
        "knn_dtw_k": KNN_DTW_K,
        "knn_dtw_agg": KNN_DTW_AGG,
        "knn_dtw_zscore": KNN_DTW_ZSCORE,
        "knn_dtw_band_frac": KNN_DTW_BAND_FRAC,
        # iforest_raw toggle
        "IFOREST_ZSCORE": IFOREST_ZSCORE,
        # LSTM AE 
        "lstm_ae": {
            "latent": LSTM_LATENT, "epochs": LSTM_EPOCHS, "batch": LSTM_BATCH,
            "val_split": LSTM_VAL_SPLIT, "patience": LSTM_PATIENCE,
            "l2": LSTM_L2, "verbose": LSTM_VERBOSE,
        },
    }

def _build_detector():
    name = DETECTOR.lower()
    # Preserve old behavior: iforest + IFOREST_ZSCORE -> use raw-series iforest
    if name == "iforest" and IFOREST_ZSCORE:
        name = "iforest_raw"
    return make_detector(name, _detector_cfg())

# τ from inner OOF (with optional resamples)
def _apply_tau_margin(tau_base: float, oof_scores: np.ndarray) -> float:
    mode = str(TAU_MARGIN_MODE).lower()
    if mode == "off":
        return float(tau_base)
    if mode == "mult":
        return float(tau_base * (1.0 + float(TAU_MARGIN_VALUE)))
    return float(tau_base)

def _tau_from_oof(X_tr: np.ndarray) -> float:
    """
    Calibrate τ from out-of-fold genuine scores on the training subset.
    Assumes detector.score: higher = more anomalous; reject if score >= τ.
    Applies a configurable safety margin on top of the base quantile.
    """
    n_tr = X_tr.shape[0]
    q = 1.0 - FRR_TARGET
    oof_scores: List[float] = []

    if n_tr < 2:
        det_tmp = _build_detector().fit(X_tr)
        oof_scores = det_tmp.score(X_tr).astype(float).tolist()
    else:
        if int(OOF_REPEATS) <= 1:
            k = max(2, min(OOF_FOLDS_MAX, n_tr))
            kf = KFold(n_splits=k, shuffle=True, random_state=RANDOM_SEED + 1)
            for tr_sub, cal_sub in kf.split(np.arange(n_tr)):
                det_fold = _build_detector().fit(X_tr[tr_sub])
                oof_scores.extend(det_fold.score(X_tr[cal_sub]).astype(float).tolist())
        else:
            rng = np.random.RandomState(RANDOM_SEED + 123)
            repeats = int(OOF_REPEATS)
            for _ in range(repeats):
                k = max(2, min(OOF_FOLDS_MAX, n_tr))
                kf = KFold(n_splits=k, shuffle=True, random_state=rng.randint(0, 2**31 - 1))
                for tr_sub, cal_sub in kf.split(np.arange(n_tr)):
                    det_fold = _build_detector().fit(X_tr[tr_sub])
                    oof_scores.extend(det_fold.score(X_tr[cal_sub]).astype(float).tolist())

    oof_arr = np.asarray(oof_scores, dtype=float)
    if oof_arr.size == 0:
        det_tmp = _build_detector().fit(X_tr)
        oof_arr = det_tmp.score(X_tr).astype(float)

    tau_base = float(np.quantile(oof_arr, q))
    tau = _apply_tau_margin(tau_base, oof_arr)
    return tau

# Outer K helper 
def _outer_k_for(n: int, train_min: int):
    gap = n - train_min
    if gap <= 0:
        return None
    return int(math.ceil(n / gap))  # ensures test size = gap and train >= train_min


def per_user_eval_nested_kfold(return_metrics: bool=False):
    np.random.seed(RANDOM_SEED)
    by_user = load_normals_by_user(BASE_PATH)

    min_needed = max(USER_MIN_SAMPLES, TRAIN_MIN_SAMPLES)
    users = sorted([u for u, Xu in by_user.items() if Xu.shape[0] >= min_needed])

    print(f"Users eligible: {len(users)} (min samples ≥ {min_needed}); "
          f"detector={DETECTOR}; global LP={'ON' if APPLY_BUTTERWORTH else 'OFF'} "
          f"({BW_CUTOFF_HZ} Hz); "
          f"concat={'ON' if CONCAT_START_IN_FRONT else 'OFF'} "
          f"(front={CONCAT_SAMPLES_AFTER_START} + cap={TOTAL_LEN}); "
          f"mag_proc={MAG_PROC_MODE}; "
          f"OOF_folds_max={OOF_FOLDS_MAX} repeats={OOF_REPEATS}; "
          f"iforest_z={'ON' if (DETECTOR.lower() in {'iforest','iforest_raw'} and IFOREST_ZSCORE) else 'OFF'}; "
          f"quant_knn_z={'ON' if KNN_QUANT_ZSCORE and DETECTOR.lower().startswith('quant_') else 'OFF'}")

    if not users:
        if return_metrics: return (np.nan, np.nan, np.nan, 0, np.nan, np.nan, np.nan, np.nan)
        raise SystemExit("No users with enough samples.")

    per_user_metrics = []
    pooled_genuine_scores = []
    pooled_impostor_scores = []

    for u in users:
        X_u_all = by_user[u]  # (N_u, C, T)
        n_u     = X_u_all.shape[0]

        k_req = _outer_k_for(n_u, TRAIN_MIN_SAMPLES)
        if k_req is None or k_req < 2:
            print(f"[user={u}] SKIP: n_u={n_u} <= TRAIN_MIN_SAMPLES={TRAIN_MIN_SAMPLES}")
            continue

        outer = KFold(n_splits=k_req, shuffle=True, random_state=RANDOM_SEED)

        outer_splits = []
        seed_u = (RANDOM_SEED + int(np.frombuffer(u.encode('utf-8'), dtype=np.uint8).sum())) % (2**32 - 1)
        for fold_id, (tr_full, te_idx) in enumerate(outer.split(np.arange(n_u)), start=1):
            rng = np.random.RandomState(seed_u + fold_id)
            tr_idx = np.sort(rng.choice(tr_full, size=TRAIN_MIN_SAMPLES, replace=False))
            outer_splits.append((tr_idx, te_idx))

        rej_cnt = 0
        te_cnt  = 0
        far_fold_vals = []
        taus = []
        k_far_tot = 0
        n_far_tot = 0

        user_genuine_scores = []
        user_impostor_scores = []

        print(f"[user={u}] n={n_u}  folds={len(outer_splits)}")

        for fold_no, (tr_idx, te_idx) in enumerate(outer_splits, start=1):
            X_tr = X_u_all[tr_idx]
            X_te = X_u_all[te_idx]

            tau_u = _tau_from_oof(X_tr)
            taus.append(tau_u)

            det = _build_detector().fit(X_tr)

            sc_g = det.score(X_te)
            user_genuine_scores.append(sc_g)
            rej_cnt += int((sc_g >= tau_u).sum())
            te_cnt  += int(sc_g.size)

            k_acc = 0
            n_acc = 0
            fold_impostor_scores = []
            for v in users:
                if v == u: continue
                sc_ip = det.score(by_user[v])
                fold_impostor_scores.append(sc_ip)
                k_acc += int((sc_ip < tau_u).sum())  # impostor accepted
                n_acc += int(sc_ip.size)
            if fold_impostor_scores:
                sc_ip_all = np.concatenate(fold_impostor_scores, axis=0)
                user_impostor_scores.append(sc_ip_all)

            k_far_tot += k_acc
            n_far_tot += n_acc
            far_fold_vals.append((k_acc / n_acc) if n_acc > 0 else np.nan)

            print(f"  - fold {fold_no}: train={TRAIN_MIN_SAMPLES}  test={len(te_idx)}  "
                  f"OOF_g={len(X_tr)}  impostors_scored={n_acc}")

        FRR_u   = (rej_cnt / te_cnt) if te_cnt > 0 else np.nan
        FAR_u   = float(np.nanmean(far_fold_vals)) if far_fold_vals else np.nan
        tau_m   = float(np.mean(taus)) if taus else np.nan

        if user_genuine_scores:
            ug = np.concatenate(user_genuine_scores, axis=0)
        else:
            ug = np.empty((0,), dtype=float)
        if user_impostor_scores:
            ui = np.concatenate(user_impostor_scores, axis=0)
        else:
            ui = np.empty((0,), dtype=float)

        eer_u, tau_eer_u, frr_at_eer_u, far_at_eer_u = compute_eer(ug, ui)

        if ug.size > 0: pooled_genuine_scores.append(ug)
        if ui.size > 0: pooled_impostor_scores.append(ui)

        per_user_metrics.append((u, te_cnt, n_far_tot, k_far_tot, FRR_u, FAR_u, tau_m, eer_u))

        frr_str = ("{:.4f}".format(FRR_u) if not np.isnan(FRR_u) else "nan")
        far_str = ("{:.4f}".format(FAR_u) if not np.isnan(FAR_u) else "nan")
        eer_str = ("{:.4f}".format(eer_u) if not np.isnan(eer_u) else "nan")


        print(f"[user={u}][nested] folds={len(outer_splits)} train={TRAIN_MIN_SAMPLES} "
              f"test_genuine={te_cnt} impostors={n_far_tot} FRR={frr_str} FAR={far_str} "
              f"τ_u(mean)={tau_m:.6f}  EER={eer_str} @ τ≈{tau_eer_u:.6f}")

    arr = np.array([[m[4], m[5], m[7]] for m in per_user_metrics], dtype=float)  # FRR, FAR, EER
    macro_frr = float(np.nanmean(arr[:,0])) if arr.size else np.nan
    macro_far = float(np.nanmean(arr[:,1])) if arr.size else np.nan
    macro_eer = float(np.nanmean(arr[:,2])) if arr.size else np.nan

    k_tot = sum(m[3] for m in per_user_metrics)
    n_tot = sum(m[2] for m in per_user_metrics)
    if n_tot > 0:
        pooled_far = k_tot / n_tot
        far_lo, far_hi = binomial_clopper_pearson(k_tot, n_tot, alpha=0.05)
    else:
        pooled_far = np.nan
        far_lo, far_hi = (np.nan, np.nan)

    if pooled_genuine_scores:
        pg = np.concatenate(pooled_genuine_scores, axis=0)
    else:
        pg = np.empty((0,), dtype=float)
    if pooled_impostor_scores:
        pi = np.concatenate(pooled_impostor_scores, axis=0)
    else:
        pi = np.empty((0,), dtype=float)
    pooled_eer, tau_eer_pool, _, _ = compute_eer(pg, pi)

    print("\n====================== SUMMARY ======================")
    print(f"Users evaluated: {len(per_user_metrics)}")
    print(f"Macro-FRR (mean over users): {macro_frr:.4f}")
    print(f"Macro-FAR (mean over users): {macro_far:.4f}")
    print(f"Macro-EER (mean over users): {macro_eer:.4f}")
    if n_tot>0:
        print(f"Pooled FAR over all impostor trials (across folds): {k_tot}/{n_tot} = {pooled_far:.4f} "
              f"(95% CI [{far_lo:.4f}, {far_hi:.4f}])")
    print(f"Pooled EER over all users' scores: {pooled_eer:.4f} @ τ≈{tau_eer_pool:.6f}")

    if return_metrics:
        # (macro_frr, macro_far, pooled_far, n_tot, far_lo, far_hi, macro_eer, pooled_eer)
        return (macro_frr, macro_far, pooled_far, n_tot, far_lo, far_hi, macro_eer, pooled_eer)

# Fancy summary for RUN_LIST 

def _print_overall_table(rows: List[tuple]):
    """
    rows: list of tuples (det, macro_frr, macro_far, pooled_far, n_tot, lo, hi, macro_eer, pooled_eer)
    """
    print("\n======================== RUN LIST SUMMARY ========================")
    print(f"{'Detector':<16} {'Macro-FRR':>10} {'Macro-FAR':>10} {'Macro-EER':>10} "
          f"{'Pooled FAR':>12} {'95% CI':>18} {'Pooled EER':>12} {'Trials':>8}")
    print("-"*110)
    for det, m_frr, m_far, p_far, n_tot, lo, hi, m_eer, p_eer in rows:
        ci = f"[{lo:.4f}, {hi:.4f}]" if not (np.isnan(lo) or np.isnan(hi)) else "[nan, nan]"
        print(f"{det:<16} {m_frr:>10.4f} {m_far:>10.4f} {m_eer:>10.4f} "
              f"{p_far:>12.4f} {ci:>18} {p_eer:>12.4f} {n_tot:>8}")
    print("-"*110)

# ---------------- Main ----------------
if __name__ == "__main__":
    if RUN_LIST:
        results = []
        for det in RUN_LIST:
            print("\n" + "="*26 + f" RUNNING detector={det} " + "="*26)
            globals()["DETECTOR"] = det
            try:
                metrics = per_user_eval_nested_kfold(return_metrics=True)
            except Exception as e:
                print(f"[SKIP] detector={det} failed: {e}")
                continue
            # Store (detector name , metrics tuple)
            results.append((det,) + metrics)
        if results:
            _print_overall_table(results)
        print("\nFinished RUN_LIST.")
    else:
        # Fallback: run the single detector specified above
        per_user_eval_nested_kfold()
