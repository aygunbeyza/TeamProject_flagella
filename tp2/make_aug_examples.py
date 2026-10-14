import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from PIL import Image
from scipy.ndimage import rotate

sl = pd.read_csv("processed/slices.csv")
gt = pd.read_csv("processed/gt.csv")
val = sl[(sl.split == "val") & (sl.has_motor == 1)]
m = val.merge(gt, on="slice_id", how="inner")
ok = m[(m.y > 0.3 * m.height) & (m.y < 0.7 * m.height) &
       (m.x > 0.3 * m.width) & (m.x < 0.7 * m.width)]
r = ok.iloc[0]
print("using slice:", r.slice_id, "motor at", r.y, r.x)

img = np.array(Image.open(r.path).convert("L")).astype(float) / 255
H = 320
cy, cx = int(r.y), int(r.x)
y0, x0 = max(cy - H, 0), max(cx - H, 0)
crop = img[y0:cy + H, x0:cx + H]
my, mx = cy - y0, cx - x0
mark = np.zeros_like(crop)
mark[my, mx] = 1.0

def where(a):
    i = np.unravel_index(np.argmax(a), a.shape)
    return i[1], i[0]

rng = np.random.default_rng(1)
flip, mflip = crop[:, ::-1], mark[:, ::-1]
rot = rotate(crop, 15, reshape=False, order=1, mode="constant", cval=0)
mrot = rotate(mark, 15, reshape=False, order=1, mode="constant", cval=0)
bri = np.clip((crop - 0.5) * 1.4 + 0.6, 0, 1)
noi = np.clip(crop + rng.normal(0, 0.1, crop.shape), 0, 1)

panels = [("Original", crop, mark),
          ("Flip", flip, mflip),
          ("Rotation (15°)", rot, mrot),
          ("Brightness / contrast\n(shown stronger)", bri, mark),
          ("Noise, U-Net only\n(shown stronger)", noi, mark)]
fig, ax = plt.subplots(1, 5, figsize=(15, 3.6))
for a, (t, im, mk) in zip(ax, panels):
    a.imshow(im, cmap="gray", vmin=0, vmax=1)
    px, py = where(mk)
    a.add_patch(Circle((px, py), 38, fill=False, edgecolor="red", lw=2))
    a.set_title(t, fontsize=13)
    a.axis("off")
plt.tight_layout()
plt.savefig("results_final/augmentation_examples.png", dpi=200)
print("saved results_final/augmentation_examples.png")
