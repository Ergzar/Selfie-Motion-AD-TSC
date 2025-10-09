import os

import json
from time import perf_counter
import numpy as np
from sklearn.model_selection import StratifiedKFold

from aeon.classification.interval_based import QUANTClassifier, RSTSF
from aeon.classification.convolution_based import MultiRocketHydraClassifier, MultiRocketClassifier
from aeon.classification.dictionary_based import MUSE
from aeon.classification.shapelet_based import RDSTClassifier
from aeon.classification.deep_learning import ResNetClassifier, InceptionTimeClassifier, LITETimeClassifier
from aeon.classification.feature_based import Catch22Classifier
from aeon.classification.hybrid import HIVECOTEV2
from sklearn.linear_model import LogisticRegression, RidgeClassifierCV

from scipy.signal import butter, filtfilt
from tsc_utils import (
    save_cm, get_event_timestamps, _load_seq, _pad_or_trim,
    compute_eer_and_tau_from_probs, eval_frr_far_from_probs,
    tau_at_target_frr, get_user_scores_from_proba, get_scores_from_estimator, mean_sd
)

import matplotlib as mpl
mpl.rcParams["figure.dpi"] = 200
mpl.rcParams["savefig.dpi"] = 600              
mpl.rcParams["pdf.fonttype"] = 42               
mpl.rcParams["ps.fonttype"] = 42
mpl.rcParams["svg.fonttype"] = "none"     
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score 

from sklearn.ensemble import RandomForestClassifier


BASE_PATH = os.path.expanduser("~/Desktop/Dataset")

CHANNELS = {
    # "lin_acc_": [0,1,2],
    "acc_":   [0,1,2],
    # "gyro_": [0,1,2],
    # "magnet_": [0,1,2],
}

# Butterworth filter (applied after channel stacking)
APPLY_BUTTERWORTH = True
BW_ORDER          = 2
BW_BTYPE          = "low"             # "low", "high", or "band"
BW_CUTOFF_HZ      = 12.5
FILTER_FS_HZ      = 50.0

# Magnetometer "unitdiff" settings
MAG_UNITDIFF_DEMEAN = True
MAG_UNITDIFF_ZSCORE = True # Not necessary

# Window around SELFIE_CAPTURE
SAMPLES_BEFORE = 50
SAMPLES_AFTER  = 150
TOTAL_LENGTH   = SAMPLES_BEFORE + SAMPLES_AFTER

# One of: "capture_only", "capture_plus_from_start", "start_only"
MODE = "capture_plus_from_start"

CAPTURE_REQUIRED = True

# Prepend N samples from the very start of each series in front of each window
PREPEND_START_SAMPLES = 0  # 0 disables

# per-user thresholding toggle (not used in thesis)
PER_USER_THRESHOLDS = False  # (used only in multiclass path) True = τᵤ per user; False = single global τ

# per-user one-vs-rest binary mode (only used in thesis with binary classifiers)
PER_USER_BINARY = False   # True → run a binary classifier per user (user vs. others). False → multiclass path.

# Outer CV params & outer repeats
K_FOLDS     = 5
N_REPEATS   = 1
RANDOM_SEED = 7

# Inner CV params
MAX_INNER_FOLDS = 3
INNER_SHUFFLE   = True
INNER_REPEATS   = 5  # set to 1 to disable inner resampling (used for threshold setting)

# Choose classifier
CLF_NAME = "catch22"    # used if CLF_LIST is None 
CLF_LIST = None  # ["RESNET", "MUSE", "CATCH22", "MRH", "QUANT", "RDST", "RSTSF"]

# Train-time target threshold 
FRR_TARGET  = 0.01

np.random.seed(RANDOM_SEED)

def cast_for_model(X, clf):
    dt = np.float64 if isinstance(clf, RDSTClassifier) else np.float32
    return np.ascontiguousarray(X, dtype=dt)

def make_clf(rep_idx, fold_idx, name=None):
    if name is None:
        name = CLF_NAME
    seed = RANDOM_SEED + 1000 * rep_idx + fold_idx
    name = name.upper()
    if name == "QUANT":
        return QUANTClassifier(random_state=seed)
    if name == "HIVECOTE":
        return HIVECOTEV2(n_jobs=-1, verbose=1, random_state=seed)
    if name == "MRH":
        return MultiRocketHydraClassifier(n_jobs=-1, random_state=seed)
    if name == "RDST":
        return RDSTClassifier(use_prime_dilations=True, n_jobs=-1, random_state=seed)
    if name == "RESNET":
        return ResNetClassifier(batch_size=32, n_epochs=150, random_state=seed)
    if name == "LITETIME":
        return LITETimeClassifier(batch_size=64, n_epochs=500, random_state=seed)
    if name == "MUSE":
        return MUSE(n_jobs=-1, support_probabilities=True, random_state=seed)
    if name == "RSTSF":
        return RSTSF(n_jobs=-1, random_state=seed)
    if name == "INCEPTION":
        return InceptionTimeClassifier(n_epochs=300, verbose=1)
    if name == "CATCH22":
        return Catch22Classifier(n_jobs=-1, random_state=seed)
    if name == "MULTIROCKET":
        return MultiRocketClassifier(estimator=RandomForestClassifier(), n_jobs=-1) # experimental
    raise ValueError(f"Unknown classifier name: {name}")

def window_capture_centered(file_path, ts_cap, ts_start):
    times, vals = _load_seq(file_path)
    if times is None:
        return None
    if ts_cap is None:
        if CAPTURE_REQUIRED:
            return None
        anchor = ts_start if ts_start is not None else times[0]
    else:
        anchor = ts_cap
    idx = int(np.clip(np.searchsorted(times, anchor), 0, len(times)))
    start = max(0, idx - SAMPLES_BEFORE)
    end   = min(len(times), idx + SAMPLES_AFTER)
    win = vals[start:end]
    return _pad_or_trim(win, TOTAL_LENGTH)

def window_from_start(file_path):
    times, vals = _load_seq(file_path)
    if times is None:
        return None
    start_idx = 0
    end_idx   = min(len(times), start_idx + TOTAL_LENGTH)
    win = vals[start_idx:end_idx]
    return _pad_or_trim(win, TOTAL_LENGTH)

def window_head_n(file_path, n):
    n = int(max(0, n))
    if n == 0:
        return []
    times, vals = _load_seq(file_path)
    if times is None:
        return None
    take = min(len(vals), n)
    head = vals[:take]
    if take < n:
        head = head + [head[-1]] * (n - take)
    return head

def _unitdiff_T3(B: np.ndarray, demean=True, zscore=True) -> np.ndarray:
    eps = 1e-8
    B = np.asarray(B, np.float32)
    if B.ndim != 2 or B.shape[1] < 3:
        raise ValueError("Magnetometer unitdiff requires tri-axial samples")
    B = B[:, :3]
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

def _apply_butterworth(arr, fs_hz, btype="low", cutoff_hz=20.0, order=2):
    A = np.asarray(arr, dtype=np.float32)
    if A.ndim == 1:
        A = A[None, :]
    nyq = 0.5 * float(fs_hz)
    if isinstance(cutoff_hz, (list, tuple, np.ndarray)):
        wn = [max(1e-6, min(0.999, float(c) / nyq)) for c in cutoff_hz]
    else:
        wn = max(1e-6, min(0.999, float(cutoff_hz) / nyq))
    b, a = butter(int(order), wn, btype=btype)
    return filtfilt(b, a, A, axis=-1).astype(np.float32)

def extract_multichannel(folder, ts_start, ts_cap):
    if MODE not in {"capture_only", "capture_plus_from_start", "start_only"}:
        raise ValueError(f"Invalid MODE={MODE}")
    use_capture = MODE in {"capture_only", "capture_plus_from_start"}
    use_start   = MODE in {"capture_plus_from_start", "start_only"}

    def _mag3(row):
        k = min(3, len(row))
        if k < 3:
            return None
        v = np.asarray(row[:3], dtype=float)
        return float(np.linalg.norm(v))

    chans = []
    for prefix, axes in CHANNELS.items():
        file_path = None
        for root, _, files in os.walk(folder):
            for fn in files:
                if fn.startswith(prefix) and fn.endswith(".json"):
                    file_path = os.path.join(root, fn)
                    break
            if file_path:
                break
        if not file_path:
            return None

        built = []
        if use_capture:
            w_cap = window_capture_centered(file_path, ts_cap, ts_start)
            if w_cap is None:
                return None
            built.append(w_cap)
        if use_start:
            w_start = window_from_start(file_path)
            if w_start is None:
                return None
            built.append(w_start)

        if PREPEND_START_SAMPLES > 0:
            head = window_head_n(file_path, PREPEND_START_SAMPLES)
            if head is None:
                return None
            built = [head + w for w in built]

        width = len(built[0][0])
        expected_len = TOTAL_LENGTH + int(max(0, PREPEND_START_SAMPLES))
        for w in built:
            if len(w) != expected_len or len(w[0]) != width:
                return None

        if prefix.lower().startswith("magnet"):
            if not axes:
                continue
            for w in built:
                B = np.array([[row[0], row[1], row[2]] for row in w], dtype=np.float32)
                dU = _unitdiff_T3(B, demean=MAG_UNITDIFF_DEMEAN, zscore=MAG_UNITDIFF_ZSCORE)
                for ax in axes:
                    if isinstance(ax, int) and ax in (0, 1, 2):
                        chans.append(dU[:, int(ax)].astype(float).tolist())
                    elif isinstance(ax, str) and ax.lower() in ("mag", "magnitude", "norm", "l2"):
                        chans.append(np.linalg.norm(dU, axis=1).astype(float).tolist())
                    else:
                        raise ValueError("CHANNELS['magnet_'] must contain 0/1/2 or 'mag'/'norm'/'l2'")
            continue

        for axis in axes:
            if isinstance(axis, str) and axis.lower() in ("mag", "magnitude", "norm", "l2"):
                for w in built:
                    mags = []
                    for row in w:
                        m = _mag3(row)
                        if m is None:
                            return None
                        mags.append(m)
                    chans.append(mags)
            else:
                if not isinstance(axis, int) or axis < 0 or axis >= width:
                    return None
                for w in built:
                    chans.append([row[axis] for row in w])

    arr = np.vstack(chans)  # (C, T)

    if APPLY_BUTTERWORTH:
        arr = _apply_butterworth(
            arr,
            fs_hz=FILTER_FS_HZ,
            btype=BW_BTYPE,
            cutoff_hz=BW_CUTOFF_HZ,
            order=BW_ORDER,
        )

    expected_T = TOTAL_LENGTH + int(max(0, PREPEND_START_SAMPLES))
    if arr.shape[1] != expected_T:
        return None
    return arr

def load_data():
    X, y = [], []
    users = sorted(d for d in os.listdir(BASE_PATH) if os.path.isdir(os.path.join(BASE_PATH, d)))
    for user in users:
        cls_path = os.path.join(BASE_PATH, user)
        for sample in os.listdir(cls_path):
            samp_path = os.path.join(cls_path, sample)
            if not os.path.isdir(samp_path):
                continue
            ts_start, ts_cap = get_event_timestamps(samp_path)
            arr = extract_multichannel(samp_path, ts_start, ts_cap)
            if arr is not None:
                X.append(arr)
                y.append(user)
    if not X:
        raise SystemExit("No valid samples loaded.")
    X = np.stack(X, axis=0)   # (N, C, T)
    y = np.array(y)
    return np.ascontiguousarray(X), y

# Utils for binary per-user path 

def _positive_proba(clf, X, positive_label=1):
    """Return P(positive) for binary classifiers."""
    if hasattr(clf, "predict_proba"):
        P = clf.predict_proba(X)
        if hasattr(clf, "classes_"):
            cls = clf.classes_
            if len(cls) == 2:
                if positive_label in cls:
                    pos_idx = int(np.where(cls == positive_label)[0][0])
                else:
                    # fallback: pick the larger class value as "positive"
                    pos_idx = int(np.argmax(cls))
                return P[:, pos_idx].astype(float)
        return P[:, -1].astype(float)
    if hasattr(clf, "decision_function"):
        d = clf.decision_function(X).astype(float)
        if d.ndim > 1:
            d = d[:, -1]
        dmin, dptp = d.min(), np.ptp(d)
        return (d - dmin) / (dptp + 1e-9)
    yhat = clf.predict(X)
    return (yhat == positive_label).astype(float)

def run_binary_per_user_model(MODEL_NAME, X, y, rng_seed_base=RANDOM_SEED):
    """Outer StratifiedKFold - inside each fold, train one binary classifier per user (new unique model trained and evaluated for every user in every outer fold)"""
    print(f"\nPER-USER BINARY MODEL: {MODEL_NAME}")

    K = K_FOLDS
    skf_outer = StratifiedKFold(n_splits=K, shuffle=True, random_state=rng_seed_base)
    fold_means = []          # (eer,frr,far) averaged over users per fold
    rep_train_list, rep_infer_list, rep_msps_list = [], [], []

    for fold_no, (train_idx, test_idx) in enumerate(skf_outer.split(X, y), start=1):
        X_tr, y_tr = X[train_idx], y[train_idx]
        X_te, y_te = X[test_idx],  y[test_idx]

        # per-user metrics for THIS fold
        per_user_stats = []
        per_user_train_sec = []
        per_user_infer_sec = []
        per_user_msps = []

        users_in_test = np.unique(y_te)
        for u in users_in_test:  # evaluate only users that appear in test fold
            # Binary labels: 1 for user u, 0 otherwise
            y_tr_bin = (y_tr == u).astype(int)
            y_te_bin = (y_te == u).astype(int)

            # Inner CV: OOF genuine scores → τᵤ at FRR_TARGET
            tau_u = 0.5  # fallback
            counts_u = int(y_tr_bin.sum())
            counts_n = int((1 - y_tr_bin).sum())
            if counts_u >= 2 and counts_n >= 2: 
                min_count = int(min(counts_u, counts_n))
                inner_splits = max(2, min(MAX_INNER_FOLDS, min_count))

                n_tr = len(X_tr)
                oof_sum = np.zeros(n_tr, dtype=np.float64)
                oof_cnt = np.zeros(n_tr, dtype=np.int32)

                for inner_rep in range(1, INNER_REPEATS + 1):
                    skf_inner = StratifiedKFold(
                        n_splits=inner_splits,
                        shuffle=INNER_SHUFFLE,
                        random_state=rng_seed_base + 100*fold_no + inner_rep
                    )
                    for itr, ival in skf_inner.split(X_tr, y_tr_bin):
                        clf_inner = make_clf(inner_rep, fold_no*10 + inner_rep, name=MODEL_NAME)
                        Xi = cast_for_model(X_tr[itr], clf_inner)
                        Xv = cast_for_model(X_tr[ival], clf_inner)
                        clf_inner.fit(Xi, y_tr_bin[itr])
                        p_pos = _positive_proba(clf_inner, Xv, positive_label=1)

                        # Accumulate only genuine (positive) OOF scores
                        mask_pos = y_tr_bin[ival] == 1
                        if np.any(mask_pos):
                            idxs = ival[mask_pos]
                            oof_sum[idxs] += p_pos[mask_pos]
                            oof_cnt[idxs] += 1

                if np.any(oof_cnt > 0):
                    oof_avg_pos = (oof_sum[oof_cnt > 0] / oof_cnt[oof_cnt > 0]).astype(float)
                    tau_u = tau_at_target_frr(oof_avg_pos, target_frr=FRR_TARGET)

            # Train on full outer-train; test on outer-test
            clf = make_clf(0, fold_no, name=MODEL_NAME)
            X_tr_cast = cast_for_model(X_tr, clf)
            X_te_cast = cast_for_model(X_te, clf)

            t0 = perf_counter()
            clf.fit(X_tr_cast, y_tr_bin)
            t1 = perf_counter()

            p0 = perf_counter()
            p_pos_te = _positive_proba(clf, X_te_cast, positive_label=1)
            p1 = perf_counter()

            train_dur = (t1 - t0)
            infer_dur = (p1 - p0)
            n_test = max(1, len(X_te_cast))
            ms_per_sample = (infer_dur / n_test) * 1000.0

            # Split genuine/impostor scores on test
            gp_te = p_pos_te[y_te_bin == 1]
            ip_te = p_pos_te[y_te_bin == 0]
            if gp_te.size == 0 or ip_te.size == 0:
                continue

            eer_test, _ = compute_eer_and_tau_from_probs(gp_te, ip_te)
            frr_tau, far_tau = eval_frr_far_from_probs(gp_te, ip_te, tau_u)

            per_user_stats.append((eer_test, frr_tau, far_tau))
            per_user_train_sec.append(train_dur)
            per_user_infer_sec.append(infer_dur)
            per_user_msps.append(ms_per_sample)

        if per_user_stats:
            arr = np.asarray(per_user_stats, dtype=float)
            eer_m, frr_m, far_m = arr.mean(axis=0).tolist()
            fold_means.append((eer_m, frr_m, far_m))

            # Timing summaries per fold (averaged over users)
            if per_user_train_sec:
                rep_train_list.append(float(np.mean(per_user_train_sec)))
                rep_infer_list.append(float(np.mean(per_user_infer_sec)))
                rep_msps_list.append(float(np.mean(per_user_msps)))

            print(f"[{MODEL_NAME} | Fold {fold_no}/{K}] users={len(per_user_stats)}  "
                  f"EER(test sweep)={eer_m:.4f}  "
                  f"| @τᵤ(FRR_train={FRR_TARGET*100:.1f}%): FRR_test={frr_m:.4f} FAR_test={far_m:.4f}  "
                  f"| Train≈{np.mean(per_user_train_sec):.2f}s/user  Verify(batch)≈{np.mean(per_user_infer_sec):.3f}s/user  "
                  f"≈ {np.mean(per_user_msps):.2f} ms/sample")
        else:
            print(f"[{MODEL_NAME} | Fold {fold_no}/{K}] No evaluable users in test.")

    # Overall summary
    if fold_means:
        fm = np.asarray(fold_means, dtype=float)
        eer_fold_m, eer_fold_s = mean_sd(fm[:, 0])
        frr_fold_m, frr_fold_s = mean_sd(fm[:, 1])
        far_fold_m, far_fold_s = mean_sd(fm[:, 2])

        print("\n=== OVERALL SUMMARY (per-user binary; fold means) ===")
        print(f"[{MODEL_NAME}] EER(test sweep): {eer_fold_m:.4f} ± {eer_fold_s:.4f}")
        print(f"[{MODEL_NAME}] TEST @τᵤ(FRR_train={FRR_TARGET*100:.1f}%): "
              f"FRR_test={frr_fold_m:.4f} ± {frr_fold_s:.4f}   FAR_test={far_fold_m:.4f} ± {far_fold_s:.4f}")

# MAIN
if __name__ == "__main__":
    chans_per_axis = 1 if MODE in {"capture_only", "start_only"} else 2
    mode_desc = {
        "capture_only": "capture-centered only",
        "capture_plus_from_start": "capture-centered + from-start",
        "start_only": "from-start only",
    }[MODE]
    chan_summary = ", ".join(f"{p}{axes} x {mode_desc}" for p, axes in CHANNELS.items())

    X, y = load_data()
    users = np.unique(y)

    # global index map for CM (indices only) -- used in multiclass path
    USERS_GLOBAL = np.array(sorted(users))
    USER_TO_IDX = {u: i for i, u in enumerate(USERS_GLOBAL)}

    C = X.shape[1]
    print(
        f"Loaded {len(X)} samples from {len(users)} users, "
        f"channels={C} ({chans_per_axis} window(s) per axis), series length={X.shape[2]} "
        f"(base TOTAL_LENGTH={TOTAL_LENGTH}, prepend={PREPEND_START_SAMPLES}, {chan_summary}); "
        f"filter={'ON' if APPLY_BUTTERWORTH else 'OFF'}; "
        f"τ-mode={'per-user' if PER_USER_THRESHOLDS else 'global'}; "
        f"binary-per-user={'ON' if PER_USER_BINARY else 'OFF'}"
    )

    models_to_run = (CLF_LIST if CLF_LIST else [CLF_NAME])

    # per-user binary branch 
    if PER_USER_BINARY:
        for MODEL_NAME in models_to_run:
            run_binary_per_user_model(MODEL_NAME, X, y)
        # Skip multiclass CM/table in this mode
        raise SystemExit(0)


    final_rows = []  # for the final across-models table

    for MODEL_NAME in models_to_run:
        print(f"\n==================== MODEL: {MODEL_NAME} ====================")

        # accumulators for CM across folds × repeats for this model
        cm_true_idx_all = []
        cm_pred_idx_all = []

        acc_fold_values = []  

        per_user_entries = []  # pooled per-user diagnostics
        train_times_sec = []
        infer_batch_times_sec = []
        per_sample_verify_ms = []

        rep_train_list = []
        rep_infer_list = []
        rep_msps_list  = []

        # collectors
        fold_level_means = []         
        per_repeat_fold_means = []    

        for rep in range(1, N_REPEATS + 1):
            print(f"\n=== Repeat {rep}/{N_REPEATS} ===")
            # Make a new split for each repeat
            skf_outer = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=RANDOM_SEED + rep)

            fold_level_means_rep = []

            for fold_no, (train_idx, test_idx) in enumerate(skf_outer.split(X, y), start=1):
                X_tr, y_tr = X[train_idx], y[train_idx]
                X_te, y_te = X[test_idx],  y[test_idx]

                counts = np.array([np.sum(y_tr == u) for u in np.unique(y_tr)])
                min_count = int(np.min(counts)) if counts.size else 0

                tau_train_global = 0.5   # fallback
                tau_by_user = {}         

                if min_count >= 2: # need at least 2 samples for inner CV. Practically all classes should have 10+ sequences
                    inner_splits = max(2, min(MAX_INNER_FOLDS, min_count))
                    n_tr = len(X_tr)
                    oof_sum = np.zeros(n_tr, dtype=np.float64)
                    oof_cnt = np.zeros(n_tr, dtype=np.int32)

                    for inner_rep in range(1, INNER_REPEATS + 1):       # Repetitions (resamples for inner tr folds)
                        skf_inner = StratifiedKFold(
                            n_splits=inner_splits,
                            shuffle=INNER_SHUFFLE,
                            random_state=RANDOM_SEED + 100*rep + 10*fold_no + inner_rep
                        )

                        for inner_fold, (itr, ival) in enumerate(skf_inner.split(X_tr, y_tr), start=1):
                            # Pass seeds to classifiers for repeatability 
                            clf_inner = make_clf(rep, fold_no*100 + inner_rep*10 + inner_fold, name=MODEL_NAME)
                            # Due to RDST issues, cast to 64 bit if using it
                            X_itr_cast = cast_for_model(X_tr[itr], clf_inner)
                            X_ival_cast = cast_for_model(X_tr[ival], clf_inner)
                            clf_inner.fit(X_itr_cast, y_tr[itr])

                            # Always prefer predict_proba if implmenetation has it (all aeon clfs should have one)
                            if hasattr(clf_inner, "predict_proba"):
                                P_val = clf_inner.predict_proba(X_ival_cast)
                                # If granularity is less than 5 in predict proba, use decision_function and normalize to range [0,1] 
                                if np.unique(np.round(P_val, 6)).size <= 5 and hasattr(clf_inner, "decision_function"):
                                    D = clf_inner.decision_function(X_ival_cast)
                                    if D.ndim == 1:
                                        D = np.stack([-D, D], axis=1)
                                    P_val = (D - D.min(axis=1, keepdims=True)) / (D.ptp(axis=1, keepdims=True) + 1e-9)
                                # if not, use label only
                            else:
                                yhat_val = clf_inner.predict(X_ival_cast)
                                P_val = np.zeros((len(yhat_val), len(clf_inner.classes_)), dtype=float)
                                for i, c in enumerate(clf_inner.classes_):
                                    P_val[:, i] = (yhat_val == c).astype(float)

                            # Dict class label : column index
                            class_to_col = {c: i for i, c in enumerate(clf_inner.classes_)} # class labels are colimns

                            # Evaluation sequences are rows
                            for row_idx, idx_tr in enumerate(ival):
                                # get the column of correspondig train label (genuine user)
                                col = class_to_col.get(y_tr[idx_tr], None) 
                                if col is None:
                                    continue
                                # For that user, sum the probas. They will be averaged to get the threshold
                                oof_sum[idx_tr] += float(P_val[row_idx, col])
                                oof_cnt[idx_tr] += 1

                    # finalize per-sample OOF genuine scores for that outer fold
                    n_tr = len(X_tr)

                    # Average the thresholds from inner folds (redundantly safely)
                    oof_avg_full = np.full(n_tr, np.nan, dtype=float)
                    valid_mask = oof_cnt > 0
                    oof_avg_full[valid_mask] = (oof_sum[valid_mask] / oof_cnt[valid_mask]).astype(float)

                    # Get _global_ threshold from all valid OOF genuine scores
                    if np.any(~np.isnan(oof_avg_full)):
                        tau_train_global = tau_at_target_frr(oof_avg_full[~np.isnan(oof_avg_full)], target_frr=FRR_TARGET)

                    # Per-user thresholding (used in thesis only for binary classifiers)
                    if PER_USER_THRESHOLDS:
                        for u in np.unique(y_tr):
                            idx_u = np.where(y_tr == u)[0]
                            scores_u = oof_avg_full[idx_u]
                            val_u = scores_u[~np.isnan(scores_u)]
                            if val_u.size >= 1:
                                tau_by_user[u] = tau_at_target_frr(val_u, target_frr=FRR_TARGET)
                            else:
                                tau_by_user[u] = tau_train_global

                # Final fit on full outer-train: evaluate on outer-test
                clf = make_clf(rep, fold_no, name=MODEL_NAME)
                X_tr_cast = cast_for_model(X_tr, clf)
                X_te_cast = cast_for_model(X_te, clf)

                t0 = perf_counter()
                clf.fit(X_tr_cast, y_tr)
                t1 = perf_counter()
                train_dur = t1 - t0
                train_times_sec.append(train_dur)

                p0 = perf_counter()
                P_te_all = clf.predict_proba(X_te_cast) if hasattr(clf, "predict_proba") else None
                p1 = perf_counter()
                pred_dur = p1 - p0
                infer_batch_times_sec.append(pred_dur)

                n_test = max(1, len(X_te_cast))
                per_sample_ms = (pred_dur / n_test) * 1000.0
                per_sample_verify_ms.append(per_sample_ms)

                # predictions for CM & accuracy
                if P_te_all is not None:
                    y_pred_labels = clf.classes_[np.argmax(P_te_all, axis=1)]
                else:
                    y_pred_labels = clf.predict(X_te_cast)

                acc_fold = accuracy_score(y_te, y_pred_labels)
                acc_fold_values.append(acc_fold)

                # store indices for CM
                true_idx = [USER_TO_IDX[lab] for lab in y_te]
                pred_idx = [USER_TO_IDX[lab] for lab in y_pred_labels]
                cm_true_idx_all.extend(true_idx)
                cm_pred_idx_all.extend(pred_idx)

                users_in_test = np.unique(y_te)
                per_user = []

                for u in users_in_test:
                    if P_te_all is not None:
                        cls_pos = np.where(clf.classes_ == u)[0]
                        if cls_pos.size == 0:
                            continue
                        scores_all = P_te_all[:, int(cls_pos[0])]
                    else:
                        yhat = clf.predict(X_te_cast)
                        scores_all = (yhat == u).astype(float)

                    mask_g = (y_te == u)
                    mask_i = ~mask_g
                    if not mask_g.any() or not mask_i.any():
                        continue

                    gp_te = scores_all[mask_g]
                    ip_te = scores_all[mask_i]

                    # capability metric: EER from test sweep
                    eer_test, _ = compute_eer_and_tau_from_probs(gp_te, ip_te)

                    # operational metric: FRR/FAR at τ (global or per-user)
                    tau_use = tau_by_user.get(u, tau_train_global) if PER_USER_THRESHOLDS else tau_train_global
                    frr_tau, far_tau = eval_frr_far_from_probs(gp_te, ip_te, tau_use)

                    per_user.append((eer_test, frr_tau, far_tau))
                    per_user_entries.append((eer_test, frr_tau, far_tau))

                if per_user:
                    arr = np.asarray(per_user, dtype=float)
                    eer_m  = float(arr[:, 0].mean())
                    frr_f_m, far_f_m = float(arr[:, 1].mean()), float(arr[:, 2].mean())

                    # store fold-level means (global) and for this repeat
                    fold_level_means.append((eer_m, frr_f_m, far_f_m))
                    fold_level_means_rep.append((eer_m, frr_f_m, far_f_m))

                    print(
                        f"[{MODEL_NAME} | Repeat {rep}/{N_REPEATS} | Fold {fold_no}/{K_FOLDS}] users={len(per_user)}  "
                        f"EER(test sweep)={eer_m:.4f}  "
                        f"| @τ({'per-user' if PER_USER_THRESHOLDS else 'global'}; FRR_train={FRR_TARGET*100:.1f}%): "
                        f"FRR_test={frr_f_m:.4f} FAR_test={far_f_m:.4f}  "
                        f"| Acc={acc_fold:.4f}  "
                        f"| Train={train_dur:.2f}s  Verify(batch)={pred_dur:.3f}s  "
                        f"≈ {per_sample_ms:.2f} ms/sample"
                    )
                else:
                    print(f"[{MODEL_NAME} | Repeat {rep}/{N_REPEATS} | Fold {fold_no}/{K_FOLDS}] No evaluable users in test.")

            # === end folds: aggregate this repeat's fold means ===
            if fold_level_means_rep:
                r = np.asarray(fold_level_means_rep, dtype=float)
                per_repeat_fold_means.append(tuple(r.mean(axis=0)))  # (eer,frr,far) per repeat

            # timing summaries per repeat
            if infer_batch_times_sec[-K_FOLDS:]:
                rep_train = float(np.mean(train_times_sec[-K_FOLDS:]))
                rep_infer = float(np.mean(infer_batch_times_sec[-K_FOLDS:]))
                rep_msps  = float(np.mean(per_sample_verify_ms[-K_FOLDS:]))
                rep_train_list.append(rep_train)
                rep_infer_list.append(rep_infer)
                rep_msps_list.append(rep_msps)

        # 
        # Model-wise summaries 
        eer_for_table = float("nan")
        frr_for_table = float("nan")
        far_for_table = float("nan")

        # A) Fold-mean summary (across all folds × repeats)
        if fold_level_means:
            fm = np.asarray(fold_level_means, dtype=float)  # (K_FOLDS*N_REPEATS, 3)
            eer_fold_m, eer_fold_s = mean_sd(fm[:, 0])
            frr_fold_m, frr_fold_s = mean_sd(fm[:, 1])
            far_fold_m, far_fold_s = mean_sd(fm[:, 2])

            print("\n=== OVERALL SUMMARY (fold means across repeats × folds) ===")
            print(f"[{MODEL_NAME}] EER(test sweep): {eer_fold_m:.4f} ± {eer_fold_s:.4f}")
            print(f"[{MODEL_NAME}] TEST  @τ({'per-user' if PER_USER_THRESHOLDS else 'global'}; "
                  f"FRR_train={FRR_TARGET*100:.1f}%): "
                  f"FRR_test={frr_fold_m:.4f} ± {frr_fold_s:.4f}   FAR_test={far_fold_m:.4f} ± {far_fold_s:.4f}")

            # Use fold-means in final table
            eer_for_table, frr_for_table, far_for_table = eer_fold_m, frr_fold_m, far_fold_m

        # B) Repeat-mean summary — one number per repeat; SD is across repeats only
        if per_repeat_fold_means:
            pr = np.asarray(per_repeat_fold_means, dtype=float)  # (N_REPEATS, 3)
            eer_rep_m, eer_rep_s = mean_sd(pr[:, 0])
            frr_rep_m, frr_rep_s = mean_sd(pr[:, 1])
            far_rep_m, far_rep_s = mean_sd(pr[:, 2])

            print("\n=== OVERALL SUMMARY (repeat means across repeats) ===")
            print(f"[{MODEL_NAME}] EER(test sweep): {eer_rep_m:.4f} ± {eer_rep_s:.4f}")
            print(f"[{MODEL_NAME}] TEST  @τ({'per-user' if PER_USER_THRESHOLDS else 'global'}; "
                  f"FRR_train={FRR_TARGET*100:.1f}%): "
                  f"FRR_test={frr_rep_m:.4f} ± {frr_rep_s:.4f}   FAR_test={far_rep_m:.4f} ± {far_rep_s:.4f}")

        # Diagnostic: pooled per-user entries (naturally wider)
        if per_user_entries:
            arr_all = np.asarray(per_user_entries, dtype=float)
            eer_pu_m, eer_pu_s = mean_sd(arr_all[:, 0])
            frr_pu_m, frr_pu_s = mean_sd(arr_all[:, 1])
            far_pu_m, far_pu_s = mean_sd(arr_all[:, 2])

            print("\n--- Diagnostic (pooled per-user entries; wider by nature) ---")
            print(f"[{MODEL_NAME}] EER(test sweep): {eer_pu_m:.4f} ± {eer_pu_s:.4f}  (pooled per-user)")
            print(f"[{MODEL_NAME}] TEST  @τ({'per-user' if PER_USER_THRESHOLDS else 'global'}; "
                  f"FRR_train={FRR_TARGET*100:.1f}%): "
                  f"FRR_test={frr_pu_m:.4f} ± {frr_pu_s:.4f}   FAR_test={far_pu_m:.4f} ± {far_pu_s:.4f}  (pooled per-user)")

        # -------- Accuracy summary (across all folds × repeats) --------
        acc_mean = float("nan")
        acc_sd   = float("nan")
        if acc_fold_values:
            acc_arr = np.asarray(acc_fold_values, dtype=float)
            acc_mean, acc_sd = mean_sd(acc_arr)
            print("\n=== OVERALL CLASSIFICATION ACCURACY ===")
            print(f"[{MODEL_NAME}] ACC (argmax): {acc_mean:.4f} ± {acc_sd:.4f}")

        # timing summaries across repeats
        def _msd(x):
            return mean_sd(np.array(x)) if x else (float("nan"), float("nan"))
        train_mean, train_std = _msd(rep_train_list)
        infer_mean, infer_std = _msd(rep_infer_list)
        msps_mean,  msps_std  = _msd(rep_msps_list)

        print("\n=== TIMING SUMMARY (mean ± SD across repeats) ===")
        print(f"Train={train_mean:.2f} ± {train_std:.2f} s/fold   "
              f"Verify(batch)={infer_mean:.3f} ± {infer_std:.3f} s/fold   "
              f"≈ {msps_mean:.2f} ± {msps_std:.2f} ms/sample")

        # Save high-quality confusion matrix for this model 
        if cm_true_idx_all and cm_pred_idx_all:
            out_stem = f"cm_{MODEL_NAME.lower()}"
            save_cm(cm_true_idx_all, cm_pred_idx_all, labels=list(USERS_GLOBAL), out_stem=out_stem)
            print(f"[{MODEL_NAME}] Confusion matrix saved to {out_stem}.pdf/.svg/.png")

        # Save concise row for the final table (include timing)
        final_rows.append({
            "model": MODEL_NAME,
            "eer_test": eer_for_table,
            "frr_test": frr_for_table,
            "far_test": far_for_table,
            "train_s_fold": train_mean,
            "infer_s_fold": infer_mean,
            "ms_per_sample": msps_mean,
            "acc": acc_mean,  
        })

    # Final across-models table 
    if final_rows:
        print("\n==================== FINAL BENCHMARK TABLE ====================")
        op_label = "τ(train, per-user)" if PER_USER_THRESHOLDS else "τ(train, global)"
        hdr = (
            f"{'Model':<14} | {'EER_test':>8} | "
            f"{'FRR_test@'+op_label:>22} | {'FAR_test@'+op_label:>22} | "
            f"{'ACC':>8} | "
            f"{'Train s/fold':>12} | {'Verify s/fold':>13} | {'ms/sample':>10}"
        )
        sep = "-" * len(hdr)
        print(hdr)
        print(sep)
        for row in final_rows:
            print(
                f"{row['model']:<14} | {row['eer_test']:8.4f} | "
                f"{row['frr_test']:22.4f} | {row['far_test']:22.4f} | "
                f"{row['acc']:8.4f} | "
                f"{row['train_s_fold']:12.2f} | {row['infer_s_fold']:13.3f} | {row['ms_per_sample']:10.2f}"
            )
