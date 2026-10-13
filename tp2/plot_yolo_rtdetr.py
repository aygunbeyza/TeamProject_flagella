#!/usr/bin/env python
"""
Result figures for YOLOv8m and RT-DETR-L, drawn from results_final/all_runs.csv (written by compare_best.py).
  yolo_results.png    F1 vs label box size, bg3 vs bg0 (pretrained | from scratch) + false positives on motor-free slices
  rtdetr_results.png  staged search (validation set): learning rate -> box size -> image size
Usage (from tp2):  python plot_yolo_rtdetr.py
"""
import argparse
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

BLUE, TEAL, RED = "#00008C", "#0A777F", "#B3261E"
COL = {"Pretrained": BLUE, "From scratch": TEAL}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


def num(pat, s):
    m = re.search(pat, str(s))
    return float(m.group(1)) if m else None


def lr_label(v):
    return f"{v:.0e}".replace("e-0", "e-").replace("e+0", "e+")


def line_panel(ax, df, xcol, xlabel, title, fmt=str):
    """one line per condition; the higher point of two series gets its label above, the lower one below"""
    xs = sorted(df[xcol].unique())
    pos = {v: i for i, v in enumerate(xs)}
    for cond, g in df.groupby("condition"):
        g = g.sort_values(xcol)
        ax.plot([pos[v] for v in g[xcol]], g["F1"], "o-", color=COL[cond], lw=2.2, ms=7, label=cond)
    for v in xs:
        pts = df[df[xcol] == v].sort_values("F1", ascending=False)
        for k, (_, r) in enumerate(pts.iterrows()):
            above = (k == 0) or r.F1 < 0.1          # near the x axis: label above, not below
            ax.annotate(f"{r.F1:.2f}", (pos[v], r.F1), textcoords="offset points",
                        xytext=(0, 8 if above else -15), ha="center", fontsize=9, color=COL[r.condition])
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels([fmt(v) for v in xs])
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_title(title, fontsize=10.5)
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="results_final/all_runs.csv")
    ap.add_argument("--out_dir", default="results_final")
    a = ap.parse_args()
    d = pd.read_csv(a.csv)
    d["lr"] = d["setting"].map(lambda s: num(r"lr0=([\d.eE+-]+)", s))
    d["box"] = d["setting"].map(lambda s: num(r"box (\d+)", s))
    d["img"] = d["setting"].map(lambda s: num(r"imgsz (\d+)", s))
    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------------- YOLO: bg3 (solid) vs bg0 (dotted red), pretrained and from scratch ----------------
    y = d[d.model == "yolo"]
    boxes = sorted(y.box.unique())
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.0), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    for k, cond in enumerate(["Pretrained", "From scratch"]):
        g3 = y[(y.condition == cond) & (y.bg == 3)].sort_values("box")
        g0 = y[(y.condition == cond) & (y.bg == 0)].sort_values("box")
        pos = {v: i for i, v in enumerate(boxes)}
        ax[k].plot([pos[v] for v in g3.box], g3.F1, "o-", color=COL[cond], lw=2.2, ms=7, label="with motor-free slices (bg3)")
        ax[k].plot([pos[v] for v in g0.box], g0.F1, "D:", color=RED, lw=1.6, ms=8, label="without motor-free slices (bg0)")
        for _, r in g3.iterrows():
            hi = not (len(g0[g0.box == r.box]) and g0[g0.box == r.box].F1.iloc[0] > r.F1)
            ax[k].annotate(f"{r.F1:.2f}", (pos[r.box], r.F1), textcoords="offset points",
                           xytext=(0, 8 if hi else -15), ha="center", fontsize=9, color=COL[cond])
        for _, r in g0.iterrows():
            hi = len(g3[g3.box == r.box]) and g3[g3.box == r.box].F1.iloc[0] < r.F1
            ax[k].annotate(f"{r.F1:.2f}", (pos[r.box], r.F1), textcoords="offset points",
                           xytext=(0, 8 if hi else -15), ha="center", fontsize=9, color=RED)
        ax[k].set_xticks(range(len(boxes)))
        ax[k].set_xticklabels([f"{int(v)}" for v in boxes])
        ax[k].set_xlim(-0.5, len(boxes) - 0.5)
        ax[k].set_ylim(0, 1.05)
        ax[k].set_xlabel("Label box size (px)")
        ax[k].set_title(f"({'ab'[k]}) YOLOv8m, {cond.lower()}: F1", fontsize=10.5)
        ax[k].grid(alpha=0.3)
        ax[k].spines[["top", "right"]].set_visible(False)
    ax[0].set_ylabel("F1 (validation set)")
    ax[0].legend(loc="lower left", frameon=False, fontsize=8.5)
    # (c) false positives on motor-free validation slices
    c = ax[2]
    w = 0.2
    series = [("Pretrained", 3, BLUE, ""), ("Pretrained", 0, BLUE, "//"), ("From scratch", 3, TEAL, ""), ("From scratch", 0, TEAL, "//")]
    for j, (cond, bg, colr, hatch) in enumerate(series):
        g = y[(y.condition == cond) & (y.bg == bg)].set_index("box")
        xs = [i + (j - 1.5) * w for i, v in enumerate(boxes) if v in g.index]
        vs = [g.loc[v, "FP_bg"] for v in boxes if v in g.index]
        c.bar(xs, vs, w, color=colr, hatch=hatch, edgecolor="white" if not hatch else "white",
              label=f"{cond.lower()}, {'bg3' if bg == 3 else 'bg0'}")
        for xx, vv in zip(xs, vs):
            c.text(xx, vv + 0.25, f"{int(vv)}", ha="center", fontsize=8.5)
    c.set_xticks(range(len(boxes)))
    c.set_xticklabels([f"{int(v)}" for v in boxes])
    c.set_xlabel("Label box size (px)")
    c.set_ylabel("False positives on motor-free\nvalidation slices")
    c.set_title("(c) False positives (solid: bg3, hatched: bg0)", fontsize=10.5)
    c.legend(loc="upper right", frameon=False, fontsize=8, ncol=2)
    c.set_ylim(0, max(y.FP_bg.max() * 1.35, 4))
    c.yaxis.set_major_locator(MaxNLocator(integer=True))
    c.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig(out / "yolo_results.png", dpi=200, bbox_inches="tight")
    plt.close()

    # ---------------- RT-DETR (staged search, chain taken from the data) ----------------
    r = d[(d.model == "rtdetr") & (d.bg == 3)]
    s1, s2, s3 = [], [], []
    for cond, g in r.groupby("condition"):
        g1 = g[(g.box == 48) & (g.img == 1024)]                    # stage 1: learning rate
        lr = g1.loc[g1.F1.idxmax(), "lr"]
        g2 = g[(g.lr == lr) & (g.img == 1024)]                     # stage 2: box size at the best lr
        box = g2.loc[g2.F1.idxmax(), "box"]
        g3 = g[(g.lr == lr) & (g.box == box)]                      # stage 3: image size
        s1.append(g1); s2.append(g2); s3.append(g3)
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.9))
    line_panel(ax[0], pd.concat(s1), "lr", "Learning rate lr0", "(a) learning rate (box 48 px, 1024 px)", lr_label)
    line_panel(ax[1], pd.concat(s2), "box", "Label box size (px)", "(b) box size (best lr0, 1024 px)", lambda v: f"{int(v)}")
    line_panel(ax[2], pd.concat(s3), "img", "Image size (px)", "(c) image size (best lr0 and box)", lambda v: f"{int(v)}")
    ax[0].set_ylabel("F1 (validation set)")
    ax[0].legend(loc="upper left", frameon=False, fontsize=9)
    r0 = d[(d.model == "rtdetr") & (d.bg == 0) & (d.condition == "Pretrained")]
    if len(r0):
        q = r0.iloc[0]
        b3 = r[(r.condition == "Pretrained") & (r.lr == q.lr) & (r.box == q.box) & (r.img == q.img)]
        ax[2].text(0.5, 0.06, f"Pretrained best setting without motor-free slices (bg0):\nF1 {q.F1:.2f}, FP {int(q.FP_bg)}"
                   + (f"   (with motor-free slices: F1 {b3.F1.iloc[0]:.2f}, FP {int(b3.FP_bg.iloc[0])})" if len(b3) else ""),
                   transform=ax[2].transAxes, ha="center", fontsize=8.5, color=RED)
    fig.suptitle("RT-DETR-L: staged parameter search", fontsize=11.5)
    plt.tight_layout()
    plt.savefig(out / "rtdetr_results.png", dpi=200, bbox_inches="tight")
    print("wrote", out / "yolo_results.png", "and", out / "rtdetr_results.png")


if __name__ == "__main__":
    main()
