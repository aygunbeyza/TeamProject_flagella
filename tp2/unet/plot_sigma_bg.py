#!/usr/bin/env python
"""
U-Net: validation F1 versus heat-map width sigma, and false positives on motor-free slices,
for training WITH (bg3) and WITHOUT (bg0) motor-free slices.
Reads  <runs_dir>/*/eval_val/metrics.json   (default: unet/runs)
Writes <out_dir>/unet_sigma_bg.png and .pdf (default: unet/results)
Usage:  python unet/plot_sigma_bg.py
"""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ap = argparse.ArgumentParser()
ap.add_argument("--runs_dir", default=str(HERE / "runs"))
ap.add_argument("--out_dir", default=str(HERE / "results"))
args = ap.parse_args()

rows = []
for f in sorted(Path(args.runs_dir).glob("*/eval_val/metrics.json")):
    m = json.loads(f.read_text())
    tag = m["run_tag"]
    t = re.search(r"sigma(\d+(?:\.\d+)?)_bg(\d+)", tag)
    if not t:
        continue
    b = m["all_slices"]["best_f1"]
    rows.append({"sigma": float(t.group(1)), "bg": int(t.group(2)),
                 "F1": b["f1"], "AP": m["all_slices"]["average_precision"],
                 "FP": b["fp_on_background_slices"]})
df = pd.DataFrame(rows).sort_values(["bg", "sigma"])
with3, without = df[df.bg == 3], df[df.bg == 0]
print(df.to_string(index=False))


BLUE, TEAL, RED = "#00008C", "#0A777F", "#B3261E"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, (a, b) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})

# (a) F1 and AP versus sigma; bg3 = blue/teal, bg0 = red
a.plot(with3.sigma, with3.F1, "o-", color=BLUE, lw=2.2, ms=7, label="F1, with motor-free slices (bg3)")
a.plot(with3.sigma, with3.AP, "s--", color=TEAL, lw=1.8, ms=6, label="AP, with motor-free slices (bg3)")
a.plot(without.sigma, without.F1, "D:", color=RED, lw=1.4, ms=9, label="F1, without motor-free slices (bg0)")
a.plot(without.sigma, without.AP, "s:", color=RED, lw=1.0, mfc="white", mec=RED, mew=1.8, ms=7,
       label="AP, without motor-free slices (bg0)")
for x, v in zip(with3.sigma, with3.F1):
    a.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9, color=BLUE)
for x, v in zip(without.sigma, without.F1):
    a.annotate(f"{v:.2f}", (x, v), textcoords="offset points", xytext=(0, -15), ha="center", fontsize=9, color=RED)
a.set_xlabel("Heat-map width $\\sigma$ (px)")
a.set_ylabel("Score (validation set)")
a.set_xticks(sorted(df.sigma.unique()))
a.set_ylim(0.4, 0.9)
a.grid(alpha=0.3)
a.legend(loc="lower right", fontsize=8, frameon=False)
a.set_title("(a) F1 and AP versus $\\sigma$", fontsize=10.5)

# (b) false positives on motor-free validation slices, for every sigma that was trained with bg0
sig = sorted(without.sigma.unique())
w3 = [with3[with3.sigma == s].FP.iloc[0] for s in sig]
w0 = [without[without.sigma == s].FP.iloc[0] for s in sig]
x = np.arange(len(sig))
b.bar(x - 0.2, w3, 0.38, color=BLUE, label="with motor-free slices (bg3)")
b.bar(x + 0.2, w0, 0.38, color=RED, label="without motor-free slices (bg0)")
for xi, v3, v0 in zip(x, w3, w0):
    b.text(xi - 0.2, v3 + 0.2, f"{int(v3)}", ha="center", fontsize=10, weight="bold")
    b.text(xi + 0.2, v0 + 0.2, f"{int(v0)}", ha="center", fontsize=10, weight="bold")
b.set_xticks(x)
b.set_xticklabels([f"{s:g}" for s in sig])
b.set_xlabel("Heat-map width $\\sigma$ (px)")
b.set_ylabel("False positives on motor-free\nvalidation slices")
b.set_ylim(0, max(w3 + w0) * 1.25)
b.legend(loc="upper left", fontsize=8, frameon=False)
b.set_title("(b) False positives", fontsize=10.5)
b.spines[["top", "right"]].set_visible(False)

plt.tight_layout()
out = Path(args.out_dir)
out.mkdir(parents=True, exist_ok=True)
plt.savefig(out / "unet_sigma_bg.png", dpi=200, bbox_inches="tight")
plt.savefig(out / "unet_sigma_bg.pdf", bbox_inches="tight")
print("wrote", out / "unet_sigma_bg.png", "and .pdf")
