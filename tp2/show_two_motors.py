import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

s = pd.read_csv("processed/slices.csv")
g = pd.read_csv("processed/gt.csv")
t = s[(s.split == "train") & (s.has_motor == 1)]
cnt = t.groupby("tomo_id").size()
cands = list(cnt[cnt == 2].index)
print("tomograms with exactly 2 motor slices (first 8):", cands[:8])
tomo = sys.argv[1] if len(sys.argv) > 1 else cands[0]
rows = t[t.tomo_id == tomo].merge(g, on="slice_id")
print("using", tomo, "with", len(rows), "motors")

fig, ax = plt.subplots(1, len(rows), figsize=(5.5 * len(rows), 5.2), squeeze=False)
for a, r in zip(ax[0], rows.itertuples()):
    img = np.array(Image.open(r.path).convert("L"))
    a.imshow(img, cmap="gray")
    a.plot(r.x, r.y, "c+", ms=26, mew=3)
    a.set_title(r.slice_id, fontsize=13)
    a.axis("off")
plt.tight_layout()
plt.savefig("results_final/tomogram_two_motors.png", dpi=170)
print("saved results_final/tomogram_two_motors.png")
