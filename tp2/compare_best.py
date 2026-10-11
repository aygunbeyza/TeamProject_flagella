#!/usr/bin/env python
"""
Final comparison for the presentation / report.

Reads every evaluated run (<model>/runs/*/eval_val/metrics.json), picks the BEST run per
(condition, model) by F1 (ties: AP) and writes to results_final/:
  best_table.md / .csv / .tex     one row per (condition, model): the best run and its numbers
  all_runs.csv                    every run, sorted
  f1_ap_bars.png                  F1 and AP of the best runs
  pr_curves_best.png              precision-recall curves of the best runs
  speed_vs_f1.png                 inference time versus F1
Condition = "From scratch" (random start; U-Net is always scratch) or "Pretrained" (COCO weights).
Only runs trained WITH motor-free slices (bg3) compete for "best" (--bg), bg0 stays in all_runs.csv.
Usage:  python compare_best.py            (from the project folder)
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config

MODELS = [("unet", "U-Net"), ("yolo", "YOLOv8m"), ("rtdetr", "RT-DETR-L")]
CONDS = ["From scratch", "Pretrained"]
COLORS = {"unet": "#1f77b4", "yolo": "#ff7f0e", "rtdetr": "#2ca02c"}


def describe(model, tag, params):
    box = re.search(r"_box(\d+)", tag)
    if model == "unet":
        s = params.get("sigma")
        return f"sigma={float(s):g}" if s is not None else tag
    b = int(box.group(1)) if box else 48
    if model == "yolo":
        return f"box {b} px"
    return f"lr0={params.get('lr0')}, box {b} px, imgsz {params.get('imgsz')}"


def load_runs():
    rows = []
    for model, _ in MODELS:
        for f in sorted((config.PROJECT_DIR / model / "runs").glob("*/eval_val/metrics.json")):
            m = json.loads(f.read_text())
            a, tr, inf = m["all_slices"], m.get("train") or {}, m.get("inference") or {}
            params = m.get("params") or {}
            tag = m["run_tag"]
            bg = re.search(r"bg(\d+)", tag)
            pretrained = False if model == "unet" else bool(params.get("pretrained", "scratch" not in tag))
            rows.append({
                "model": model, "condition": "Pretrained" if pretrained else "From scratch",
                "setting": describe(model, tag, params), "bg": int(bg.group(1)) if bg else None,
                "F1": a["best_f1"]["f1"], "AP": a["average_precision"],
                "P": a["best_f1"]["precision"], "R": a["best_f1"]["recall"],
                "F1@0.5": a["fixed_threshold"]["f1"], "FP_bg": a["best_f1"]["fp_on_background_slices"],
                "err_px": a["median_localisation_error_px_at_best_f1"],
                "train_min": tr.get("train_time_min"), "fwd_ms": inf.get("forward_ms_mean"),
                "e2e_ms": inf.get("end_to_end_ms_mean"), "params_M": (tr.get("model_params") or np.nan) / 1e6,
                "run_tag": tag, "sweep": str(f.parent / "threshold_sweep.csv")})
    return pd.DataFrame(rows)


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def save(name, out, rect=(0, 0, 1, 1)):
    plt.tight_layout(rect=rect)
    plt.savefig(out / name, dpi=160)
    plt.close()
    print("wrote", out / name)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bg", type=int, default=3, help="only runs with this many motor-free slices compete for 'best'")
    ap.add_argument("--out_dir", default=str(config.PROJECT_DIR / "results_final"))
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    df = load_runs()
    if df.empty:
        sys.exit("no evaluated runs found")
    df.drop(columns="sweep").sort_values(["condition", "model", "F1"], ascending=[True, True, False]) \
        .round(4).to_csv(out / "all_runs.csv", index=False)

    best = []
    for cond in CONDS:
        for model, name in MODELS:
            d = df[(df["model"] == model) & (df["condition"] == cond) & (df["bg"] == args.bg)]
            if d.empty:
                best.append({"condition": cond, "model": model, "name": name, "setting": "-"})
                continue
            r = d.sort_values(["F1", "AP"], ascending=False).iloc[0].to_dict()
            r["name"] = name
            best.append(r)
    b = pd.DataFrame(best)

    show = b[["condition", "name", "setting", "F1", "AP", "P", "R", "F1@0.5", "FP_bg", "err_px",
              "train_min", "fwd_ms", "params_M"]].copy()
    show = show.rename(columns={"name": "model", "P": "precision", "R": "recall", "F1@0.5": "F1@thr0.5",
                                "FP_bg": "FP (no-motor)", "err_px": "median err (px)", "train_min": "train (min)",
                                "fwd_ms": "forward (ms)", "params_M": "params (M)"})
    num = [c for c in show.columns if c not in ("condition", "model", "setting")]
    show[num] = show[num].astype(float).round(3)
    show.to_csv(out / "best_table.csv", index=False)
    (out / "best_table.md").write_text("# Best run per model and starting condition (selected by F1 on val)\n\n"
                                       + md_table(show) + "\n")
    print("\n" + show.to_string(index=False))

    cols = ["condition", "model", "setting", "F1", "AP", "precision", "recall", "FP (no-motor)", "train (min)", "forward (ms)"]
    tex = ["\\begin{tabular}{lllrrrrrrr}", "\\toprule",
           "Condition & Model & Best setting & F1 & AP & Prec. & Rec. & FP$_{bg}$ & Train (min) & Fwd (ms) \\\\", "\\midrule"]
    for _, r in show[cols].iterrows():
        fmt = {"F1": "{:.3f}", "AP": "{:.3f}", "precision": "{:.2f}", "recall": "{:.2f}",
               "FP (no-motor)": "{:.0f}", "train (min)": "{:.1f}", "forward (ms)": "{:.1f}"}
        vals = ["--" if (isinstance(v, float) and np.isnan(v)) else (fmt[c].format(v) if c in fmt else str(v))
                for c, v in zip(cols, r.values)]
        vals[2] = vals[2].replace("_", "\\_")
        tex.append(" & ".join(vals) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out / "best_table.tex").write_text("\n".join(tex) + "\n")

    have = b[b["F1"].notna()].reset_index(drop=True)

    # ---- F1 / AP bars ----
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for k, (metric, title) in enumerate([("F1", "F1 (best threshold)"), ("AP", "Average precision")]):
        for i, (model, name) in enumerate(MODELS):
            for j, cond in enumerate(CONDS):
                r = have[(have["model"] == model) & (have["condition"] == cond)]
                if r.empty:
                    continue
                v = float(r[metric].iloc[0])
                x = i + (j - 0.5) * 0.38
                ax[k].bar(x, v, 0.36, color=COLORS[model], alpha=0.45 if cond == "From scratch" else 1.0,
                          hatch="//" if cond == "From scratch" else None, edgecolor="white")
                ax[k].text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=8)
        ax[k].set_xticks(range(len(MODELS)))
        ax[k].set_xticklabels([n for _, n in MODELS])
        ax[k].set_title(title, fontsize=10)
        ax[k].set_ylim(0, 1.05)
        ax[k].grid(axis="y", alpha=0.3)
    ax[0].set_ylabel("score on validation set")
    from matplotlib.patches import Patch
    fig.legend(handles=[Patch(facecolor="grey", alpha=0.45, hatch="//", label="From scratch"),
                        Patch(facecolor="grey", label="Pretrained (COCO)")], loc="upper center", ncol=2, fontsize=9)
    save("f1_ap_bars.png", out, rect=(0, 0, 1, 0.93))

    # ---- PR curves of the best runs ----
    plt.figure(figsize=(6.2, 5))
    for _, r in have.iterrows():
        sw = pd.read_csv(df.loc[(df["model"] == r["model"]) & (df["run_tag"] == r["run_tag"]), "sweep"].iloc[0])
        plt.plot(sw["recall"], sw["precision"], ".-" if r["condition"] == "Pretrained" else ".--",
                 color=COLORS[r["model"]], alpha=1.0 if r["condition"] == "Pretrained" else 0.6,
                 label=f"{r['name']} ({'pretrained' if r['condition'] == 'Pretrained' else 'scratch'}), F1={r['F1']:.2f}")
    plt.xlabel("recall"); plt.ylabel("precision"); plt.xlim(0, 1.02); plt.ylim(0, 1.02)
    plt.grid(alpha=0.3); plt.legend(fontsize=7, loc="lower left")
    plt.title("Precision-recall curves of the best runs (each point = one score threshold)", fontsize=9)
    save("pr_curves_best.png", out)

    # ---- speed vs F1 ----
    plt.figure(figsize=(6, 4.5))
    for _, r in have.iterrows():
        plt.scatter(r["fwd_ms"], r["F1"], s=40 + 8 * float(r["params_M"]), color=COLORS[r["model"]],
                    alpha=1.0 if r["condition"] == "Pretrained" else 0.5, edgecolor="black", linewidth=0.5)
        plt.annotate(f"{r['name']} ({'pre' if r['condition'] == 'Pretrained' else 'scr'})", (r["fwd_ms"], r["F1"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=7)
    plt.xlabel("forward pass per image (ms, lower is faster)"); plt.ylabel("F1")
    plt.grid(alpha=0.3); plt.title("Accuracy versus inference time (marker size = parameters)", fontsize=9)
    save("speed_vs_f1.png", out)


if __name__ == "__main__":
    main()
