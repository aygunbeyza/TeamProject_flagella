"""
Model-independent evaluation: object matching, F-beta, precision-recall sweep.

Every detector (U-Net, YOLO, RT-DETR) only has to write a predictions file with
columns  slice_id, y, x, score  (original image pixel coordinates).  This module
does the rest, so all models are scored by exactly the same code.

Matching rule
-------------
Within one slice the detections are processed from highest to lowest score.
Each detection is matched to the nearest still-unmatched ground-truth motor; it
is a true positive if that distance (converted to physical units with the
slice's voxel spacing) is <= hit distance.  Otherwise it is a false positive.
Unmatched ground truths are false negatives.

Because detections are processed in score order, whether a detection is a TP
does NOT depend on the score threshold.  Matching is therefore done once, and
the threshold sweep only counts.
"""
import numpy as np
import pandas as pd


def fbeta(precision, recall, beta=1.0):
    b2 = beta ** 2
    denom = b2 * precision + recall
    return (1 + b2) * precision * recall / denom if denom > 0 else 0.0


def match_detections(slices, gt, pred, hit_nm):
    """
    Returns one row per detection with columns
    slice_id, y, x, score, is_tp, dist_px, on_background_slice.
    """
    spacing = dict(zip(slices["slice_id"], slices["voxel_spacing"]))
    has_motor = dict(zip(slices["slice_id"], slices["has_motor"]))
    unknown = set(pred["slice_id"]) - set(spacing)
    if unknown:
        raise ValueError(f"predictions contain {len(unknown)} slice_id(s) not in the split, "
                         f"e.g. {sorted(unknown)[:3]}")

    gt_by_slice = {sid: g[["y", "x"]].to_numpy(float) for sid, g in gt.groupby("slice_id")}
    rows = []
    for sid, g in pred.groupby("slice_id"):
        g = g.sort_values("score", ascending=False)
        gts = gt_by_slice.get(sid, np.zeros((0, 2)))
        used = np.zeros(len(gts), dtype=bool)
        sp = float(spacing[sid])
        for r in g.itertuples():
            is_tp, dist = False, np.nan
            if len(gts):
                d = np.hypot(gts[:, 0] - r.y, gts[:, 1] - r.x)
                d_free = np.where(used, np.inf, d)
                j = int(np.argmin(d_free))
                if np.isfinite(d_free[j]) and d_free[j] * sp <= hit_nm:
                    is_tp, dist = True, float(d[j])
                    used[j] = True
            rows.append((sid, r.y, r.x, r.score, is_tp, dist, has_motor[sid] == 0))
    return pd.DataFrame(rows, columns=["slice_id", "y", "x", "score", "is_tp",
                                       "dist_px", "on_background_slice"])


def sweep_thresholds(det, n_gt, thresholds):
    """Precision / recall / F1 / F2 for every score threshold."""
    out = []
    for t in thresholds:
        kept = det[det["score"] >= t]
        tp = int(kept["is_tp"].sum())
        fp = len(kept) - tp
        fn = n_gt - tp
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / n_gt if n_gt else 0.0
        out.append({"threshold": float(t), "precision": p, "recall": r,
                    "f1": fbeta(p, r, 1.0), "f2": fbeta(p, r, 2.0),
                    "tp": tp, "fp": fp, "fn": fn,
                    "fp_on_background_slices": int((~kept["is_tp"] & kept["on_background_slice"]).sum()),
                    "fp_on_motor_slices": int((~kept["is_tp"] & ~kept["on_background_slice"]).sum())})
    return pd.DataFrame(out)


def average_precision(det, n_gt):
    """Area under the precision-recall curve obtained by ranking ALL detections by score."""
    if n_gt == 0 or len(det) == 0:
        return 0.0
    d = det.sort_values("score", ascending=False).reset_index(drop=True)
    tp_cum = d["is_tp"].cumsum().to_numpy()
    precision = tp_cum / (np.arange(len(d)) + 1)
    recall = tp_cum / n_gt
    prev = np.concatenate([[0.0], recall[:-1]])
    return float(np.sum((recall - prev) * precision))


def summarize_subset(det, n_gt, n_slices, n_bg_slices, thresholds, fixed_threshold):
    """All headline numbers for one subset of slices."""
    sweep = sweep_thresholds(det, n_gt, thresholds)
    best_f1 = sweep.loc[sweep["f1"].idxmax()]
    best_f2 = sweep.loc[sweep["f2"].idxmax()]
    fixed = sweep.loc[(sweep["threshold"] - fixed_threshold).abs().idxmin()]

    def pack(row):
        return {k: (int(row[k]) if k in ("tp", "fp", "fn", "fp_on_background_slices",
                                         "fp_on_motor_slices") else float(row[k]))
                for k in ("threshold", "precision", "recall", "f1", "f2", "tp", "fp", "fn",
                          "fp_on_background_slices", "fp_on_motor_slices")}

    kept_best = det[det["score"] >= best_f1["threshold"]]
    tp_dist = kept_best.loc[kept_best["is_tp"], "dist_px"]
    fp_bg = float(best_f1["fp_on_background_slices"])
    return {
        "n_slices": int(n_slices), "n_background_slices": int(n_bg_slices), "n_gt_motors": int(n_gt),
        "average_precision": average_precision(det, n_gt),
        "best_f1": pack(best_f1),
        "best_f2": pack(best_f2),
        "fixed_threshold": pack(fixed),
        "false_alarms_per_background_slice_at_best_f1": (fp_bg / n_bg_slices) if n_bg_slices else None,
        "median_localisation_error_px_at_best_f1": float(tp_dist.median()) if len(tp_dist) else None,
    }, sweep
