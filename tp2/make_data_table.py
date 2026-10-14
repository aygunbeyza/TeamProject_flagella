import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

s = json.load(open("processed/dataset_stats.json"))
tr, va = s["train"], s["val"]
bg = s["bg_per_tomo"]
NAVY, ROW_A, ROW_B = "#00008C", "#E8ECFB", "#F5F7FE"
hdr = ["", "Tomograms", "Slices with\na motor", "Motor-free\nslices", "Total\nslices"]
rows = [
    [f"Train (bg{bg})", tr["tomograms"], tr["motor_slices"], tr["background_slices"],
     tr["motor_slices"] + tr["background_slices"]],
    ["Train (bg0)", tr["tomograms"], tr["motor_slices"], 0, tr["motor_slices"]],
    ["Validation", va["tomograms"], va["motor_slices"], va["background_slices"],
     va["motor_slices"] + va["background_slices"]],
]
rows = [[str(c) for c in r] for r in rows]
widths = [2.2, 1.7, 1.9, 1.9, 1.5]
rh, hh = 0.62, 0.85
W = sum(widths); H = hh + rh * len(rows)
fig = plt.figure(figsize=(W, H))
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")
x = 0
for w, t in zip(widths, hdr):
    ax.add_patch(Rectangle((x + 0.02, 0.02), w - 0.04, hh - 0.04, color=NAVY, lw=0))
    ax.text(x + w / 2, hh / 2, t, color="white", ha="center", va="center", fontsize=14, weight="bold", linespacing=1.2)
    x += w
for i, r in enumerate(rows):
    y = hh + i * rh
    x = 0
    for j, (w, t) in enumerate(zip(widths, r)):
        ax.add_patch(Rectangle((x + 0.02, y + 0.02), w - 0.04, rh - 0.04, color=ROW_A if i % 2 == 0 else ROW_B, lw=0))
        ax.text(x + w / 2, y + rh / 2, t, ha="center", va="center", fontsize=14,
                weight="bold" if j in (0, 4) else "normal", color=NAVY if j == 0 else "black")
        x += w
fig.savefig("results_final/data_table_bg0_bg3.png", dpi=220, facecolor="white")
for r in rows:
    print(r)
print("wrote results_final/data_table_bg0_bg3.png")
