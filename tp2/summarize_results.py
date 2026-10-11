#!/usr/bin/env python
"""
Collect every evaluated experiment into one comparison (tables, plots, text summary).
Usage: python summarize_results.py --runs_dirs unet/runs --out_dir unet/results

Scans  <runs_dir>/*/eval_<split>/  for every given runs dir (unet/runs, later also yolo/runs rtdetr/runs)
and writes into --out_dir :

  all_results.csv              one row per experiment (all numbers)
  comparison.md                readable table (copy into report / slides)
  summary.txt                  plain-text summary: best run per metric + short ranking
  results_overview.json        same numbers as json

  pr_curves.png                precision-recall curves, all experiments in one plot
  curves_vs_threshold.png      precision / recall / F1 / F2 versus score threshold, all experiments
  metrics_bar_comparison.png   grouped bars: F1, F2, precision, recall, AP per experiment
  metrics_vs_sigma.png         F1 / F2 / precision / recall / AP versus sigma (per model, per bg setting)
  f2_vs_sigma.png              F2 versus sigma on its own (the Kaggle metric)
  false_positives.png          false positives on background slices at the best-F1 threshold
  cost_comparison.png          training time and inference speed (if available)

Use --model / --contains to restrict, e.g.  --model unet   or  --contains bg3
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config

METRICS = [("F1", "F1"), ("F2", "F2"), ("P@F1", "precision"), ("R@F1", "recall"), ("AP", "AP")]


def parse_tag(tag):
    s = re.search(r"sigma(\d+)", tag)
    b = re.search(r"bg(\d+)", tag)
    return (float(s.group(1)) if s else np.nan), (int(b.group(1)) if b else None)


def short(row):
    return f"{row['model']} s{int(row['sigma'])} bg{row['bg']}" if not np.isnan(row["sigma"]) else f"{row['model']}/{row['run_tag']}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", default="val")
    ap.add_argument("--runs_dirs", nargs="+", default=[str(config.MODEL_DIRS["unet"] / "runs")],
                    help="one or more runs folders, e.g. unet/runs yolo/runs rtdetr/runs")
    ap.add_argument("--out_dir", default=str(config.MODEL_DIRS["unet"] / "results"))
    ap.add_argument("--model", default=None, help="only this model folder (unet / yolo / rtdetr)")
    ap.add_argument("--contains", default=None, help="only run tags containing this text")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows, curves = [], {}
    files = sorted(f for d in args.runs_dirs for f in Path(d).glob(f"*/eval_{args.split}/metrics.json"))
    for f in files:
        m = json.loads(f.read_text())
        if args.model and m["model"] != args.model:
            continue
        if args.contains and args.contains not in m["run_tag"]:
            continue
        a, mo = m["all_slices"], m["motor_slices_only"]
        tr, inf = m.get("train") or {}, m.get("inference") or {}
        sigma, bg = parse_tag(m["run_tag"])
        row = {
            "model": m["model"], "run_tag": m["run_tag"], "sigma": sigma, "bg": bg,
            "AP": a["average_precision"],
            "F1": a["best_f1"]["f1"], "F1_thr": a["best_f1"]["threshold"],
            "P@F1": a["best_f1"]["precision"], "R@F1": a["best_f1"]["recall"],
            "F2": a["best_f2"]["f2"], "F2_thr": a["best_f2"]["threshold"],
            "P@F2": a["best_f2"]["precision"], "R@F2": a["best_f2"]["recall"],
            "F1@0.5": a["fixed_threshold"]["f1"], "F2@0.5": a["fixed_threshold"]["f2"],
            "FP_background@F1": a["best_f1"]["fp_on_background_slices"],
            "FP_per_bg_slice@F1": a["false_alarms_per_background_slice_at_best_f1"],
            "F1_motor_only": mo["best_f1"]["f1"], "F2_motor_only": mo["best_f2"]["f2"],
            "median_err_px": a["median_localisation_error_px_at_best_f1"],
            "epochs": tr.get("epochs"), "train_min": tr.get("train_time_min"),
            "fwd_ms": inf.get("forward_ms_mean"), "e2e_ms": inf.get("end_to_end_ms_mean"),
            "n_val_slices": a["n_slices"], "n_gt": a["n_gt_motors"],
        }
        row["label"] = short(row)
        rows.append(row)
        sweep = f.parent / "threshold_sweep.csv"
        if sweep.exists():
            curves[row["label"]] = pd.read_csv(sweep)

    if not rows:
        print(f"no evaluated runs found under {args.runs_dirs} (run evaluate.py first)")
        return
    df = pd.DataFrame(rows).sort_values(["model", "bg", "sigma", "run_tag"]).reset_index(drop=True)
    df.drop(columns="label").round(4).to_csv(out / "all_results.csv", index=False)
    (out / "results_overview.json").write_text(df.drop(columns="label").round(4).to_json(orient="records", indent=2))
    labels = list(df["label"])
    colors = {l: plt.cm.tab10(i % 10) for i, l in enumerate(labels)}

    # ------------------------------------------------------------------ comparison.md
    show = df[["model", "run_tag", "F1", "F2", "P@F1", "R@F1", "AP", "FP_background@F1", "epochs", "train_min", "fwd_ms"]]
    md = ["# Results overview", "",
          f"Validation split `{args.split}` (motor + background slices, identical for every model), hit distance "
          f"{config.HIT_DISTANCE_PX_AT_REF} px at reference spacing {config.REFERENCE_SPACING} ({config.HIT_DISTANCE_NM:.1f} physical units).",
          "F1 / F2 use the best threshold on the validation set (slightly optimistic).", "",
          "| " + " | ".join(show.columns) + " |", "|" + "---|" * len(show.columns)]
    for _, r in show.iterrows():
        md.append("| " + " | ".join("-" if pd.isna(v) else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r) + " |")
    md += ["", "P@F1 / R@F1 = precision / recall at the best-F1 threshold; FP_background@F1 = false positives on background "
           "slices at that threshold; fwd_ms = network forward pass per image.", ""]
    (out / "comparison.md").write_text("\n".join(md))

    # ------------------------------------------------------------------ plots
    def save(name):
        plt.tight_layout()
        plt.savefig(out / name, dpi=130)
        plt.close()

    # PR curves
    plt.figure(figsize=(7.5, 6))
    for l, sw in curves.items():
        plt.plot(sw["recall"], sw["precision"], ".-", color=colors[l], label=l)
    plt.xlabel("recall"); plt.ylabel("precision"); plt.xlim(0, 1.02); plt.ylim(0, 1.02)
    plt.grid(alpha=0.3); plt.legend(fontsize=7)
    plt.title("Precision-recall curves (each point = one score threshold)", fontsize=10)
    save("pr_curves.png")

    # metrics versus threshold
    fig, ax = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for a_, key, title in zip(ax.ravel(), ["precision", "recall", "f1", "f2"], ["precision", "recall", "F1", "F2"]):
        for l, sw in curves.items():
            a_.plot(sw["threshold"], sw[key], ".-", color=colors[l], label=l)
        a_.set_title(title); a_.set_ylim(0, 1.02); a_.grid(alpha=0.3)
    for a_ in ax[1]:
        a_.set_xlabel("score threshold")
    ax[0, 0].legend(fontsize=7)
    fig.suptitle("Metrics versus score threshold (all validation slices)", fontsize=11)
    save("curves_vs_threshold.png")

    # grouped bars
    fig, ax = plt.subplots(figsize=(max(8, 1.4 * len(df) + 4), 5))
    w = 0.8 / len(METRICS)
    x = np.arange(len(df))
    for k, (col, name) in enumerate(METRICS):
        bars = ax.bar(x + k * w - 0.4 + w / 2, df[col], w, label=name)
        for b_, v in zip(bars, df[col]):
            ax.text(b_.get_x() + b_.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", fontsize=6, rotation=90)
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.25); ax.grid(axis="y", alpha=0.3); ax.legend(ncol=5, fontsize=8, loc="upper center")
    ax.set_title("F1, F2, precision, recall (at the best-F1 threshold) and AP per experiment")
    save("metrics_bar_comparison.png")

    # metrics versus sigma
    sig = df.dropna(subset=["sigma"])
    if sig["sigma"].nunique() > 1:
        groups = sig.groupby(["model", "bg"])
        fig, ax = plt.subplots(1, len(METRICS), figsize=(4 * len(METRICS), 4), sharey=True)
        for a_, (col, name) in zip(ax, METRICS):
            for (model, bg), g in groups:
                g = g.sort_values("sigma")
                a_.plot(g["sigma"], g[col], "o-", label=f"{model} bg{bg}")
            a_.set_title(name); a_.set_xlabel("sigma"); a_.grid(alpha=0.3); a_.set_ylim(0, 1.02)
        ax[0].legend(fontsize=8)
        fig.suptitle("Effect of the heat-map sigma", fontsize=11)
        save("metrics_vs_sigma.png")

        plt.figure(figsize=(6, 4.5))
        for (model, bg), g in groups:
            g = g.sort_values("sigma")
            plt.plot(g["sigma"], g["F2"], "o-", label=f"{model} bg{bg}")
            for xx, yy in zip(g["sigma"], g["F2"]):
                plt.annotate(f"{yy:.2f}", (xx, yy), textcoords="offset points", xytext=(0, 6), ha="center", fontsize=7)
        plt.xlabel("sigma"); plt.ylabel("F2 (best threshold)"); plt.ylim(0, 1.02); plt.grid(alpha=0.3); plt.legend(fontsize=8)
        plt.title("F2 versus sigma")
        save("f2_vs_sigma.png")

    # false positives on background
    plt.figure(figsize=(max(6, 0.9 * len(df) + 3), 4.5))
    plt.bar(labels, df["FP_background@F1"], color=[colors[l] for l in labels])
    plt.xticks(rotation=30, ha="right", fontsize=8); plt.ylabel("false positives on background slices")
    plt.title("False alarms on motor-free slices (at best-F1 threshold)"); plt.grid(axis="y", alpha=0.3)
    save("false_positives.png")

    # cost
    cost = df.dropna(subset=["train_min", "fwd_ms"], how="all")
    if len(cost):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        ax[0].bar(cost["label"], cost["train_min"].fillna(0)); ax[0].set_ylabel("training time [min]")
        ax[1].bar(cost["label"], cost["fwd_ms"].fillna(0)); ax[1].set_ylabel("forward pass [ms / image]")
        for a_ in ax:
            a_.tick_params(axis="x", rotation=30, labelsize=8); a_.grid(axis="y", alpha=0.3)
        fig.suptitle("Cost: training time and inference speed")
        save("cost_comparison.png")

    # ------------------------------------------------------------------ summary.txt
    def best(col, lower=False):
        r = df.loc[df[col].idxmin() if lower else df[col].idxmax()]
        return r, f"{r['label']} ({col} = {r[col]:.3f})"

    L = ["=" * 78, "SUMMARY OF ALL EXPERIMENTS", "=" * 78, "",
         f"{len(df)} experiments on the {args.split} split ({int(df['n_val_slices'].iloc[0])} slices, {int(df['n_gt'].iloc[0])} ground-truth motors). "
         f"Training and validation use motor AND background slices unless the tag says bg0.", ""]
    L += ["BEST EXPERIMENT PER METRIC",
          "  F1                 : " + best("F1")[1],
          "  F2                 : " + best("F2")[1],
          "  precision @best F1 : " + best("P@F1")[1],
          "  recall @best F1    : " + best("R@F1")[1],
          "  AP                 : " + best("AP")[1],
          "  fewest FP on background slices : " + best("FP_background@F1", lower=True)[1], ""]
    L += ["RANKING BY F2 (Kaggle metric)"]
    for i, (_, r) in enumerate(df.sort_values("F2", ascending=False).iterrows(), 1):
        L.append(f"  {i:2d}. {r['label']:<24s} F2={r['F2']:.3f} (thr {r['F2_thr']:.2f})  F1={r['F1']:.3f}  "
                 f"P={r['P@F1']:.3f}  R={r['R@F1']:.3f}  AP={r['AP']:.3f}  FP_bg={int(r['FP_background@F1'])}")
    L.append("")
    if sig["sigma"].nunique() > 1:
        spread = sig["F2"].max() - sig["F2"].min()
        L += ["SIGMA", f"  F2 ranges from {sig['F2'].min():.3f} to {sig['F2'].max():.3f} over sigma = "
              f"{', '.join(str(int(s)) for s in sorted(sig['sigma'].unique()))} (spread {spread:.3f}).",
              f"  With {int(df['n_gt'].iloc[0])} validation motors one motor changes recall by {100 / df['n_gt'].iloc[0]:.1f} points, "
              "so small differences between sigma values are not statistically meaningful (single seed).", ""]
    if (df["bg"] == 0).any() and (df["bg"] > 0).any():
        L += ["BACKGROUND SLICES IN TRAINING (bg0 = motor slices only)"]
        for sg, g in df.dropna(subset=["sigma"]).groupby("sigma"):
            if set(g["bg"]) >= {0} and len(set(g["bg"])) > 1:
                a0, a3 = g[g["bg"] == 0].iloc[0], g[g["bg"] > 0].sort_values("bg").iloc[-1]
                L.append(f"  sigma {int(sg)}: FP on background slices {int(a0['FP_background@F1'])} (bg0) vs {int(a3['FP_background@F1'])} (bg{a3['bg']}), "
                         f"F2 {a0['F2']:.3f} vs {a3['F2']:.3f}")
        L.append("")
    L += ["FILES", "  all_results.csv, comparison.md, pr_curves.png, curves_vs_threshold.png, metrics_bar_comparison.png,",
          "  metrics_vs_sigma.png, f2_vs_sigma.png, false_positives.png, cost_comparison.png",
          "  per-experiment details: runs/<model>/<run_tag>/eval_val/summary.txt, pr_curve.png and loss_curve.png",
          "  NOTE: thresholds were chosen on the validation set -> numbers are slightly optimistic.", "=" * 78]
    (out / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print("\n".join(md))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
