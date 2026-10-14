import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import rotate

sl = pd.read_csv("processed/slices.csv")
gt = pd.read_csv("processed/gt.csv")
print("gt columns:", list(gt.columns))
val = sl[(sl.split == "val") & (sl.has_motor == 1)]
m = val.merge(gt, on="slice_id", how="inner", suffixes=("", "_gt"))
ycol = "y" if "y" in m.columns else [c for c in m.columns if c.endswith("y")][0]
xcol = "x" if "x" in m.columns else [c for c in m.columns if c.endswith("x")][0]
# motor well inside the image (not at the border)
ok = m[(m[ycol] > 0.3 * m.height) & (m[ycol] < 0.7 * m.height) &
       (m[xcol] > 0.3 * m.width) & (m[xcol] < 0.7 * m.width)]
r = ok.iloc[0]
print("using slice:", r.slice_id, "motor at", r[ycol], r[xcol])

img = np.array(Image.open(r.path).convert("L")).astype(float) / 255
H = 320
cy, cx = int(r[ycol]), int(r[xcol])
crop = img[max(cy - H, 0):cy + H, max(cx - H, 0):cx + H]

rng = np.random.default_rng(1)
flip = crop[:, ::-1]
rot = rotate(crop, 15, reshape=False, mode="reflect")
bri = np.clip(crop * 1.2 + 0.1, 0, 1)
noi = np.clip(crop + rng.normal(0, 0.05, crop.shape), 0, 1)

panels = [("Original", crop), ("Flip", flip), ("Rotation (15°)", rot),
          ("Brightness", bri), ("Noise (U-Net only)", noi)]
fig, ax = plt.subplots(1, 5, figsize=(15, 3.4))
for a, (t, im) in zip(ax, panels):
    a.imshow(im, cmap="gray", vmin=0, vmax=1)
    a.set_title(t, fontsize=14)
    a.axis("off")
plt.tight_layout()
plt.savefig("results_final/augmentation_examples.png", dpi=200)
print("saved results_final/augmentation_examples.png")
