#!/usr/bin/env python
"""
Result figures for YOLOv8m and RT-DETR-L, drawn from results_final/all_runs.csv (written by compare_best.py).
  yolo_results.png    (a) F1 vs label box size, pretrained / from scratch (bg3)
                      (b) pretrained, box 48: with (bg3) vs without (bg0) motor-free training slices
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

    # ---------------- YOLO ----------------
    y = d[d.model == "yolo"]
    fig, (p, q) = plt.subplots(1, 2, figsize=(11, 4.0), gridspec_kw={"width_ratios": [1.6, 1]})
    line_panel(p, y[y.bg == 3], "box", "Label box size (px)", "(a) YOLOv8m: label box size", lambda v: f"{int(v)}")
    p.set_ylabel("F1 (validation set)")
    p.legend(loc="lower right", frameon=False, fontsize=9)
    ref = y[(y.condition == "Pretrained") & (y.box == 48)]
    ref = ref[ref.bg.isin([0, 3])].sort_values("bg", ascending=False)      # bg3 first, bg0 second
    if len(ref) == 2:
        q.bar([0, 1], ref["F1"], 0.55, color=[BLUE, RED])
        for i, (_, r) in enumerate(ref.iterrows()):
            q.text(i, r.F1 + 0.02, f"F1 {r.F1:.2f}\nFP: {int(r.FP_bg)}", ha="center", fontsize=10, weight="bold")
        q.set_xticks([0, 1])
        q.set_xticklabels(["with motor-free\nslices (bg3)", "without motor-free\nslices (bg0)"])
        q.set_ylim(0, 1.15)
        q.set_ylabel("F1 (validation set)")
        q.spines[["top", "right"]].set_visible(False)
    q.set_title("(b) Pretrained, box 48 px:\nmotor-free training slices", fontsize=10.5)
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
    fig.suptitle("RT-DETR-L: staged parameter search", fontsize=11.5)
    plt.tight_layout()
    plt.savefig(out / "rtdetr_results.png", dpi=200, bbox_inches="tight")
    print("wrote", out / "yolo_results.png", "and", out / "rtdetr_results.png")


if __name__ == "__main__":
    main()
