import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

NAVY, RED = "#00008C", "#A93226"
d = pd.read_csv("results_final/all_runs.csv")
d = d[(d.model == "rtdetr") & (d.condition == "Pretrained")].copy()
d["lr"] = d.setting.str.extract(r"lr0=([0-9.e+-]+)")[0].astype(float)
d["box"] = d.setting.str.extract(r"box (\d+)")[0].astype(int)
d["img"] = d.setting.str.extract(r"imgsz (\d+)")[0].astype(int)
d = d[d.img == 1024]
cfg = [(5e-5, 48), (1e-4, 48), (2e-4, 48), (4e-4, 48), (2e-4, 32), (2e-4, 64)]
def lab(lr, box):
    return f"lr {lr:.0e}".replace("e-0", "e-") + f"\nbox {box}"
rows = []
for lr, box in cfg:
    row = {"lab": lab(lr, box)}
    for bg in (3, 0):
        g = d[(d.bg == bg) & (d.lr == lr) & (d.box == box)]
        row[f"F1_{bg}"] = g.F1.iloc[0] if len(g) else float("nan")
        row[f"FP_{bg}"] = g.FP_bg.iloc[0] if len(g) else float("nan")
    rows.append(row)
t = pd.DataFrame(rows)
print(t.round(3).to_string(index=False))
fig, ax = plt.subplots(1, 2, figsize=(12, 3.9))
w = 0.38
xs = range(len(t))
for k, (col, ttl, yl) in enumerate([("F1", "(a) F1 (validation set)", "F1"),
                                    ("FP", "(b) False positives on motor-free slices", "False positives")]):
    a = ax[k]
    for off, bg, colr, hatch, name in [(-w / 2, 3, NAVY, "", "with motor-free slices (bg3)"),
                                       (w / 2, 0, RED, "//", "without motor-free slices (bg0)")]:
        v = t[f"{col}_{bg}"]
        a.bar([x + off for x in xs], v, w, color=colr, hatch=hatch, edgecolor="white", label=name)
        for x, vv in zip(xs, v):
            if pd.notna(vv):
                a.text(x + off, vv + (0.015 if col == "F1" else 0.6), f"{vv:.2f}" if col == "F1" else f"{int(vv)}",
                       ha="center", fontsize=8.5)
    a.set_xticks(list(xs)); a.set_xticklabels(t.lab, fontsize=9)
    a.set_title(ttl, fontsize=10.5); a.set_ylabel(yl)
    a.spines[["top", "right"]].set_visible(False)
    if col == "F1": a.set_ylim(0, 1.25)
    else:
        a.set_ylim(0, max(t[["FP_3", "FP_0"]].max().max() * 1.15, 4))
        a.yaxis.set_major_locator(MaxNLocator(integer=True))
ax[0].legend(loc="upper left", frameon=False, fontsize=8.5)
fig.suptitle("RT-DETR-L (pretrained, 1024 px): with vs. without motor-free training slices", fontsize=11.5)
plt.tight_layout()
plt.savefig("results_final/rtdetr_bg0_vs_bg3.png", dpi=200, bbox_inches="tight")
print("wrote results_final/rtdetr_bg0_vs_bg3.png")
