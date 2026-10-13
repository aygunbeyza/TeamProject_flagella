import os
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

CSV = "results_final/all_runs.csv"
OUT = "results_final/best_comparison_table.png"
NAMES = {"yolo": "YOLOv8m", "rtdetr": "RT-DETR-L", "unet": "U-Net"}
ORDER = {"yolo": 0, "unet": 1, "rtdetr": 2}
NAVY, ROW_A, ROW_B, SEP = "#00008C", "#E8ECFB", "#F5F7FE", "#C9D0F0"

d = pd.read_csv(CSV)
d = d[d["bg"] == 3]                                   # best only among runs with motor-free slices (bg3)
best = d.loc[d.groupby(["condition", "model"])["F1"].idxmax()].copy()
best["o"] = best["model"].map(ORDER)
best["c"] = best["condition"].map({"Pretrained": 0, "From scratch": 1})
best = best.sort_values(["c", "o"])

hdr = ["Setting", "Model", "F1", "AP", "FP", "Error (px)"]
widths = [2.0, 2.0, 1.2, 1.2, 1.0, 1.5]
rows, bold = [], []
for cond, g in best.groupby("c", sort=True):
    top = g["F1"].max()
    for _, r in g.iterrows():
        rows.append([r["condition"], NAMES[r["model"]], f"{r['F1']:.3f}", f"{r['AP']:.3f}",
                     f"{int(r['FP_bg'])}", f"{r['err_px']:.2f}"])
        bold.append(r["F1"] == top)

rh, hh = 0.6, 0.55
W = sum(widths); H = hh + rh * len(rows)
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
x = 0
for w, t in zip(widths, hdr):
    ax.add_patch(Rectangle((x + 0.02, 0.02), w - 0.04, hh - 0.04, color=NAVY, lw=0))
    ax.text(x + w / 2, hh / 2, t, color="white", ha="center", va="center", fontsize=15, weight="bold")
    x += w
prev = None
for i, r in enumerate(rows):
    y = hh + i * rh
    x = 0
    for j, (w, t) in enumerate(zip(widths, r)):
        ax.add_patch(Rectangle((x + 0.02, y + 0.02), w - 0.04, rh - 0.04,
                               color=ROW_A if r[0] == "Pretrained" else ROW_B, lw=0))
        show = t if not (j == 0 and r[0] == prev) else ""
        wt = "bold" if (j == 0 or (bold[i] and j in (1, 2))) else "normal"
        ax.text(x + w / 2, y + rh / 2, show, ha="center", va="center", fontsize=14, weight=wt)
        x += w
    prev = r[0]
os.makedirs("results_final", exist_ok=True)
fig.savefig(OUT, dpi=200, facecolor="white")
print(best[["condition", "model", "setting", "F1", "AP", "FP_bg", "err_px"]].to_string(index=False))
print("->", OUT)
