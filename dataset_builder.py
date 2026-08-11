"""
Roadmap step 4: 2D dataset returning (image, heatmap, is_pos).

Positives : slices that contain a motor -> Gaussian heatmap at (y=axis1, x=axis2)
Negatives : random slices from tomograms with Number of motors == 0 -> all-zero heatmap
            (meeting notes 08.07: "1st thing to try: add negative images to the dataset")
"""

import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------- paths / defaults
BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
CSV_PATH = os.path.join(BASE_DIR, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")

DEFAULT_SIGMA = 6.0    # meeting notes: sigma = 6 px, try increasing further later
DEFAULT_PATCH = 512    # meeting notes: full-sized images can be tried later (patch_size=None)
DEFAULT_NEG_PER_TOMO = 3


# ---------------------------------------------------------------- helpers
def load_and_normalize(path):
    """Roadmap step 1: clip to 0.5/99.5 percentiles, scale to [0, 1]."""
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def gaussian_heatmap(h, w, y, x, sigma):
    """Same-size heatmap, zero everywhere except a 2D Gaussian centered on (y, x)."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2.0 * sigma ** 2)).astype(np.float32)


# ---------------------------------------------------------------- sample list
def build_samples(tomo_ids, neg_per_tomo=DEFAULT_NEG_PER_TOMO, seed=0, verbose=True):
    """
    Build the list of samples for the given tomo_ids (train or val list from step 3).

    Returns
    -------
    samples : list of (img_path, y, x, is_pos)
        is_pos == 1 -> motor at (y, x)
        is_pos == 0 -> no motor, y = x = -1
    n_pos, n_neg : int
    """
    rng = np.random.default_rng(seed)
    df = pd.read_csv(CSV_PATH)
    df = df[df["tomo_id"].isin(tomo_ids)]

    samples = []

    # ---- POSITIVES: the slice that contains the motor ----
    for _, r in df[df["Number of motors"] > 0].iterrows():
        z = int(r["Motor axis 0"])
        path = os.path.join(TRAIN_DIR, r["tomo_id"], f"slice_{z:04d}.jpg")
        if os.path.exists(path):
            # axis 1 = row = y, axis 2 = column = x
            samples.append((path, int(r["Motor axis 1"]), int(r["Motor axis 2"]), 1))
    n_pos = len(samples)

    # ---- NEGATIVES: rows with -1,-1,-1 and Number of motors == 0 ----
    neg_ids = df.loc[df["Number of motors"] == 0, "tomo_id"].unique()
    for tid in neg_ids:
        d = os.path.join(TRAIN_DIR, tid)
        if not os.path.isdir(d):
            continue
        slices = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
        if not slices:
            continue
        k = min(neg_per_tomo, len(slices))
        for s in rng.choice(slices, size=k, replace=False):
            samples.append((os.path.join(d, s), -1, -1, 0))
    n_neg = len(samples) - n_pos

    if verbose:
        ratio = n_neg / max(1, len(samples))
        print(f"[build_samples] pos={n_pos}  neg={n_neg}  total={len(samples)}  "
              f"(neg ratio {ratio:.2f})")
    return samples, n_pos, n_neg


# ---------------------------------------------------------------- dataset
class MotorSliceDataset(Dataset):
    """Returns (image[1,H,W], heatmap[1,H,W], is_pos)."""

    def __init__(self, samples, sigma=DEFAULT_SIGMA, patch_size=DEFAULT_PATCH, seed=None):
        self.samples = samples

        self.sigma = sigma
        self.ps = patch_size          # None -> use the full slice
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y, x, is_pos = self.samples[idx]
        img = load_and_normalize(path)
        h, w = img.shape

        if self.ps and self.ps < min(h, w):
            if is_pos == 1:
                # crop window guaranteed to contain the motor
                t_lo, t_hi = max(0, y - self.ps + 1), min(h - self.ps, y)
                l_lo, l_hi = max(0, x - self.ps + 1), min(w - self.ps, x)
                t = int(self.rng.integers(t_lo, max(t_lo, t_hi) + 1))
                l = int(self.rng.integers(l_lo, max(l_lo, l_hi) + 1))
                y, x = y - t, x - l     # coordinates relative to the crop
            else:
                # negatives: no motor to center on -> fully random crop
                t = int(self.rng.integers(0, h - self.ps + 1))
                l = int(self.rng.integers(0, w - self.ps + 1))
            img = img[t:t + self.ps, l:l + self.ps]

        ph, pw = img.shape
        if is_pos == 1:
            hm = gaussian_heatmap(ph, pw, y, x, self.sigma)
        else:
            hm = np.zeros((ph, pw), dtype=np.float32)

        return (torch.from_numpy(np.ascontiguousarray(img)).unsqueeze(0),
                torch.from_numpy(hm).unsqueeze(0),
                int(is_pos))


# ---------------------------------------------------------------- self-check
if __name__ == "__main__":
    with open(os.path.join(BASE_DIR, "train_ids.txt")) as f:
        train_ids = {line.strip() for line in f if line.strip()}

    samples, n_pos, n_neg = build_samples(train_ids)
    ds = MotorSliceDataset(samples, seed=0)

    pos_idx = next(i for i, s in enumerate(samples) if s[3] == 1)
    neg_idx = next(i for i, s in enumerate(samples) if s[3] == 0)

    for name, i in (("positive", pos_idx), ("negative", neg_idx)):
        img, hm, is_pos = ds[i]
        print(f"{name:9s} img={tuple(img.shape)} hm_max={float(hm.max()):.3f} is_pos={is_pos}")
