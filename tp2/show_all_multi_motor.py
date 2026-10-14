import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

s = pd.read_csv("processed/slices.csv")
g = pd.read_csv("processed/gt.csv")
t = s[s.has_motor == 1].merge(g, on="slice_id")
cnt = t.groupby("tomo_id").size()
multi = cnt[cnt >= 2].sort_values(ascending=False)
print("tomograms with 2+ motor slices:", len(multi), "(train + validation)")

PER_PAGE, W = 8, 400
for p, start in enumerate(range(0, len(multi), PER_PAGE), 1):
    ids = list(multi.index[start:start + PER_PAGE])
    ncol = max(int(multi[i]) for i in ids)
    fig, ax = plt.subplots(len(ids), ncol, figsize=(3.2 * ncol, 3.4 * len(ids)), squeeze=False)
    for i, tomo in enumerate(ids):
        rows = list(t[t.tomo_id == tomo].sort_values("slice_id").itertuples())
        for j in range(ncol):
            a = ax[i][j]
            a.axis("off")
            if j < len(rows):
                r = rows[j]
                im = Image.open(r.path).convert("L")
                f = W / max(im.size)
                im = im.resize((int(im.size[0] * f), int(im.size[1] * f)))
                a.imshow(np.array(im), cmap="gray")
                a.plot(r.x * f, r.y * f, "c+", ms=14, mew=2)
                a.set_title(r.slice_id + " (" + r.split + ")", fontsize=9)
    plt.tight_layout()
    out = "results_final/multi_motor_tomograms_p%d.png" % p
    plt.savefig(out, dpi=100)
    plt.close()
    print("saved", out, "with", len(ids), "tomograms,", ncol, "columns")
