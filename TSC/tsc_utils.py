import os
# (optional) improve determinism before heavy imports
os.environ.setdefault("PYTHONHASHSEED", "0")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
from time import perf_counter
import numpy as np
from sklearn.model_selection import StratifiedKFold


def save_cm(y_true_idx, y_pred_idx, labels=None, out_stem="cm",
            annotate=True, font_size=6, cell_min_px=18):
    """
    Save a confusion matrix heatmap to PNG/SVG/PDF.

    Parameters
    ----------
    y_true_idx : array-like of int
        True class indices (0..N-1).
    y_pred_idx : array-like of int
        Predicted class indices (0..N-1).
    labels : sequence or None
        If provided, length defines N (= number of classes). Values are *not*
        drawn; only indices are shown on axes to avoid names in the figure.
    out_stem : str
        Output filename stem without extension (e.g., "cm_catch22").
    annotate : bool
        Draw counts in cells.
    font_size : int
        Font size of annotations.
    cell_min_px : int
        Minimum pixel size per cell (figure is scaled accordingly).
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from sklearn.metrics import confusion_matrix

    y_true_idx = np.asarray(y_true_idx, dtype=int)
    y_pred_idx = np.asarray(y_pred_idx, dtype=int)

    # Determine number of classes
    if labels is not None:
        n_users = int(len(labels))
    else:
        n_users = int(max(y_true_idx.max(initial=-1), y_pred_idx.max(initial=-1)) + 1)

    # Build CM using indices 0..N-1 (we intentionally do NOT pass names)
    cm = confusion_matrix(y_true_idx, y_pred_idx, labels=list(range(n_users)))

    # Figure size so each cell is at least `cell_min_px` px
    tmp_fig = plt.figure()
    dpi = tmp_fig.dpi
    plt.close(tmp_fig)
    w_in = max(6.0, (n_users * cell_min_px) / dpi)
    h_in = max(5.0, (n_users * cell_min_px) / dpi)

    fig, ax = plt.subplots(figsize=(w_in, h_in))
    im = ax.imshow(cm, interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xlabel("Predicted (index)", fontsize=13)
    ax.set_ylabel("True (index)", fontsize=13)
    ax.set_xticks(range(n_users)); ax.set_yticks(range(n_users))
    idx_ticks = [str(i) for i in range(n_users)]
    ax.set_xticklabels(idx_ticks, rotation=90, fontsize=9)
    ax.set_yticklabels(idx_ticks, fontsize=9)

    if annotate and cm.size:
        norm = Normalize(vmin=cm.min(), vmax=cm.max() if cm.max() > 0 else 1.0)
        for i in range(n_users):
            for j in range(n_users):
                val = cm[i, j]
                # Auto-contrast text color based on normalized luminance
                c = "white" if norm(val) > 0.5 else "black"
                ax.text(j, i, f"{val}", ha="center", va="center", fontsize=font_size, color=c)

    fig.tight_layout()
    # Save high quality exports
    fig.savefig(f"{out_stem}.png")
    fig.savefig(f"{out_stem}.svg")
    fig.savefig(f"{out_stem}.pdf")
    plt.close(fig)

def get_event_timestamps(folder):
    touch = os.path.join(folder, "touch")
    if not os.path.isdir(touch):
        return None, None
    jf = next((f for f in os.listdir(touch) if f.endswith(".json")), None)
    if not jf:
        return None, None
    try:
        with open(os.path.join(touch, jf), encoding="utf-8") as f:
            events = json.load(f)
    except Exception:
        return None, None
    ts_start = next((e.get("timestamp") for e in events if e.get("variant") == "SELFIE_START"), None)
    ts_cap   = next((e.get("timestamp") for e in events if e.get("variant") == "SELFIE_CAPTURE"), None)
    return ts_start, ts_cap

def _load_seq(path):
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    seq = [(d.get("timestampMillis"), d.get("values"))
           for d in data
           if "timestampMillis" in d and "values" in d]
    seq = [(t, v) for (t, v) in seq if isinstance(v, list) and len(v) > 0]
    if not seq:
        return None, None
    seq.sort(key=lambda x: x[0])
    times = [t for t, _ in seq]
    vals  = [v for _, v in seq]
    return times, vals

def _pad_or_trim(window, total_len):
    if window is None or not window:
        return None
    if len(window) < total_len:
        window = window + [window[-1]] * (total_len - len(window))
    elif len(window) > total_len:
        window = window[:total_len]
    return window

def compute_eer_and_tau_from_probs(genuine_scores, impostor_scores):
    gp = np.asarray(genuine_scores, dtype=float)
    ip = np.asarray(impostor_scores, dtype=float)
    if gp.size == 0 or ip.size == 0:
        return 0.5, 0.5
    vals = np.unique(np.concatenate([gp, ip, [0.0, 1.0]]))
    taus = np.concatenate(([0.0 - 1e-12], (vals[:-1] + vals[1:]) / 2, [1.0 + 1e-12]))
    FRR = np.array([(gp <  t).mean() for t in taus])
    FAR = np.array([(ip >= t).mean() for t in taus])
    diff = FAR - FRR
    eq = np.where(diff == 0)[0]
    if eq.size > 0:
        i = int(eq[0]); return float(FRR[i]), float(taus[i])
    sgn = np.sign(diff)
    cross = np.where(sgn[:-1] * sgn[1:] < 0)[0]
    if cross.size > 0:
        i = int(cross[0])
        x0, x1 = taus[i], taus[i+1]
        y0, y1 = diff[i], diff[i+1]
        w = abs(y0) / (abs(y0) + abs(y1))
        tau = float(x0 * (1 - w) + x1 * w)
        eer = float(((FRR[i] * (1 - w) + FRR[i+1] * w) + (FAR[i] * (1 - w) + FAR[i+1] * w)) / 2.0)
        return eer, tau
    j = int(np.lexsort(((FAR + FRR), np.abs(diff)))[0])
    eer = float(0.5 * (FAR[j] + FRR[j]))
    tau = float(taus[j])
    return eer, tau

def eval_frr_far_from_probs(genuine_scores, impostor_scores, tau):
    gp = np.asarray(genuine_scores, dtype=float)
    ip = np.asarray(impostor_scores, dtype=float)
    if gp.size == 0 or ip.size == 0:
        return 0.5, 0.5
    FRR = float((gp <  tau).mean())
    FAR = float((ip >= tau).mean())
    return FRR, FAR

def tau_at_target_frr(genuine_scores, target_frr=0.10):
    gp = np.sort(np.asarray(genuine_scores, dtype=float))
    if gp.size == 0:
        return 0.5
    q = float(np.clip(target_frr, 0.0, 1.0))
    idx = int(np.floor(q * (gp.size - 1)))
    idx = np.clip(idx, 0, gp.size - 1)
    return float(gp[idx])

def get_user_scores_from_proba(P, y_true, classes, user_label):
    cls_pos = np.where(classes == user_label)[0]
    if cls_pos.size == 0:
        return np.array([]), np.array([])
    col = int(cls_pos[0])
    s = P[:, col]
    mask_g = (y_true == user_label)
    return s[mask_g], s[~mask_g]

def get_scores_from_estimator(clf, X, y_true, classes):
    if hasattr(clf, "predict_proba"):
        P = clf.predict_proba(X)
        if np.unique(np.round(P, 6)).size <= 5 and hasattr(clf, "decision_function"):
            D = clf.decision_function(X)
            if D.ndim == 1:
                D = np.stack([-D, D], axis=1)
            D = (D - D.min(axis=1, keepdims=True)) / (D.ptp(axis=1, keepdims=True) + 1e-9)
            P = D
    else:
        yhat = clf.predict(X)
        P = np.zeros((len(yhat), len(clf.classes_)), dtype=float)
        for i, c in enumerate(clf.classes_):
            P[:, i] = (yhat == c).astype(float)

    gp_pool, ip_pool = [], []
    for u in clf.classes_:
        gp, ip = get_user_scores_from_proba(P, y_true, clf.classes_, u)
        if gp.size: gp_pool.append(gp)
        if ip.size: ip_pool.append(ip)
    gp_pool = np.concatenate(gp_pool) if gp_pool else np.array([])
    ip_pool = np.concatenate(ip_pool) if ip_pool else np.array([])
    return gp_pool, ip_pool

def mean_sd(a):
    a = np.asarray(a, dtype=float)
    if a.size == 0:
        return 0.0, 0.0
    if a.size == 1:
        return float(a[0]), 0.0
    return float(np.mean(a)), float(np.std(a, ddof=1))
