import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

s = pd.read_csv("processed/slices.csv")
g = pd.read_csv("processed/gt.csv")
t = s[(s.split == "train") & (s.has_motor == 1)]
cnt = t.groupby("tomo_id").size()
if len(sys.argv) > 1:
    tomos = sys.argv[1:]
else:
    tomos = ["tomo_381add", "tomo_0de3ee", cnt[cnt == 3].index[0]]
print("tomograms:", tomos)

data = []
for tomo in tomos:
    r = t[t.tomo_id == tomo].merge(g, on="slice_id").sort_values("slice_id")
    data.append(list(r.itertuples()))
ncol = max(len(d) for d in data)
fig, ax = plt.subplots(len(data), ncol, figsize=(4.4 * ncol, 4.4 * len(data)), squeeze=False)
for i, rows in enumerate(data):
    for j in range(ncol):
        a = ax[i][j]
        a.axis("off")
        if j < len(rows):
            r = rows[j]
            img = np.array(Image.open(r.path).convert("L"))
            a.imshow(img, cmap="gray")
            a.plot(r.x, r.y, "c+", ms=22, mew=3)
            a.set_title(r.slice_id, fontsize=12)
plt.tight_layout()
plt.savefig("results_final/tomograms_multi_motor.png", dpi=140)
print("saved results_final/tomograms_multi_motor.png")
