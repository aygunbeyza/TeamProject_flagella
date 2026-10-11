"""Image helpers shared by every model (numpy / scipy / PIL only, no torch)."""
import numpy as np
from PIL import Image
from scipy.ndimage import zoom as ndi_zoom


def load_normalized(path, lo_pct=0.5, hi_pct=99.5):
    """Grey-scale image, clipped to the 0.5 / 99.5 percentiles and scaled to [0, 1]."""
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    lo, hi = np.percentile(img, lo_pct), np.percentile(img, hi_pct)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0).astype(np.float32)


def rescale_to_reference(img, voxel_spacing, reference_spacing):
    """
    Pixel-size normalisation: resample so that one pixel always corresponds to
    `reference_spacing`.

    Returns (image, scale_y, scale_x).  A point (y, x) of the original image
    lies at (y * scale_y, x * scale_x) in the rescaled image, and a point
    (y', x') of the rescaled image lies at (y' / scale_y, x' / scale_x) in the
    original image.
    """
    if not voxel_spacing or not reference_spacing or np.isnan(voxel_spacing):
        return img, 1.0, 1.0
    s = float(voxel_spacing) / float(reference_spacing)
    if abs(s - 1.0) < 1e-3:
        return img, 1.0, 1.0
    out = ndi_zoom(img, s, order=1).astype(np.float32)
    return out, out.shape[0] / img.shape[0], out.shape[1] / img.shape[1]


def gaussian_heatmap(h, w, points, sigma):
    """Heat map that is the pixel-wise maximum of one 2-D Gaussian per (y, x) point."""
    hm = np.zeros((h, w), dtype=np.float32)
    ys = np.arange(h, dtype=np.float32)[:, None]
    xs = np.arange(w, dtype=np.float32)[None, :]
    for (py, px) in points:
        g = np.exp(-((ys - py) ** 2 + (xs - px) ** 2) / (2.0 * sigma ** 2))
        hm = np.maximum(hm, g.astype(np.float32))
    return hm
