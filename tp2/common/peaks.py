"""Heat map -> list of detections (peak finding + non-maximum suppression)."""
import numpy as np
from scipy.ndimage import maximum_filter


def detect_peaks(heatmap, min_score=0.05, min_distance=20):
    """
    Local maxima of `heatmap` with score >= min_score, followed by a greedy
    non-maximum suppression (a peak is dropped if a stronger kept peak is
    closer than `min_distance` pixels).

    Returns a list of (y, x, score) sorted by descending score.
    """
    local_max = maximum_filter(heatmap, size=min_distance)
    ys, xs = np.where((heatmap == local_max) & (heatmap >= min_score))
    scores = heatmap[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]
    kept = []
    for i in range(len(ys)):
        if all(np.hypot(ys[i] - ys[j], xs[i] - xs[j]) >= min_distance for j in kept):
            kept.append(i)
    return [(int(ys[i]), int(xs[i]), float(scores[i])) for i in kept]
