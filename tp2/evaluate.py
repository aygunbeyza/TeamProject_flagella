#!/usr/bin/env python
"""
EVALUATION (MODEL INDEPENDENT: U-Net, YOLO, RT-DETR all use this file).
Usage: python evaluate.py --run_dir unet/runs/<run_tag>

Input : <model>/runs/<run_tag>/predictions_<split>.csv   (slice_id, y, x, score; original pixels)
Output: <model>/runs/<run_tag>/eval_<split>/
          metrics.json          headline numbers (all slices  +  motor slices only)
          threshold_sweep.csv   precision / recall / F1 / F2 for every score threshold
          pr_curve.png          precision-recall curve (each point = one threshold), metrics vs
                                threshold, and where the false positives come from
          summary.txt           human-readable one-page summary of the whole experiment

Metric definition (identical for all models, see common/metrics.py):
  * a detection is a hit if its distance to an unmatched ground-truth motor is
    <= HIT_DISTANCE_NM  (= 24 px at the reference voxel spacing, converted per slice
    with that slice's own voxel spacing)
  * the validation set contains motor slices AND background slices (fixed in prepare_data.py)
  * "all" = every validation slice;  "motor_only" = only slices containing a motor
    (no background slice -> no chance for false alarms there)
  * F1 and F2 (beta = 2, the Kaggle metric) are both reported; thresholds are chosen on the
    validation set itself, so these numbers are slightly optimistic. The result at the
    fixed threshold (0.5) is reported as well.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from common.metrics import match_detections, summarize_subset


def fmt(row):
    return (f"thr={row['threshold']:.2f}  P={row['precision']:.3f}  R={row['recall']:.3f}  "
            f"F1={row['f1']:.3f}  F2={row['f2']:.3f}  TP={row['tp']} FP={row['fp']} FN={row['fn']} "
            f"(FP on background slices: {row['fp_on_background_slices']})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="e.g. unet/runs/sigma12_bg3_ep40_seed0")
    ap.add_argument("--split", default="val")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    model = run_dir.resolve().parent.parent.name
    out_dir = run_dir / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    slices = pd.read_csv(config.SLICES_CSV)
    slices = slices[slices["split"] == args.split].reset_index(drop=True)
    gt = pd.read_csv(config.GT_CSV)
    gt = gt[gt["slice_id"].isin(slices["slice_id"])]
    pred_path = run_dir / f"predictions_{args.split}.csv"
    pred = pd.read_csv(pred_path)
    for col in ("slice_id", "y", "x", "score"):
        if col not in pred.columns:
            raise ValueError(f"{pred_path} must have columns slice_id,y,x,score (missing '{col}')")
    pred = pred[pred["score"] >= config.MIN_SCORE]

    det = match_detections(slices, gt, pred, config.HIT_DISTANCE_NM)

    n_gt = len(gt)
    n_bg = int((slices["has_motor"] == 0).sum())
    motor_ids = set(slices.loc[slices["has_motor"] == 1, "slice_id"])
    res_all, sweep_all = summarize_subset(det, n_gt, len(slices), n_bg, config.THRESHOLDS, config.FIXED_THRESHOLD)
    res_mot, sweep_mot = summarize_subset(det[det["slice_id"].isin(motor_ids)], n_gt, len(motor_ids), 0,
                                          config.THRESHOLDS, config.FIXED_THRESHOLD)

    # ---- context: config / training / timing of this experiment (if available) ----
    def load(name):
        p = run_dir / name
        return json.loads(p.read_text()) if p.exists() else None
    cfg, train, timing = load("config.json"), load("train_summary.json"), load("inference_time.json")

    metrics = {"model": model, "run_tag": run_dir.name, "split": args.split,
               "hit_distance_nm": config.HIT_DISTANCE_NM,
               "hit_distance_px_at_reference": config.HIT_DISTANCE_PX_AT_REF,
               "reference_spacing": config.REFERENCE_SPACING,
               "all_slices": res_all, "motor_slices_only": res_mot,
               "train": train, "inference": timing,
               "params": (cfg or {}).get("params"), "data": (cfg or {}).get("data")}
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    sweep_all.round(4).to_csv(out_dir / "threshold_sweep.csv", index=False)
    sweep_mot.round(4).to_csv(out_dir / "threshold_sweep_motor_only.csv", index=False)

    # ---- figure ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].plot(sweep_all["recall"], sweep_all["precision"], "b.-", label="all val slices")
    ax[0].plot(sweep_mot["recall"], sweep_mot["precision"], "g.--", alpha=0.6, label="motor slices only")
    b = res_all["best_f1"]
    ax[0].plot(b["recall"], b["precision"], "r*", ms=14, label=f"best F1 (thr {b['threshold']:.2f})")
    ax[0].set_xlabel("recall"); ax[0].set_ylabel("precision")
    ax[0].set_xlim(0, 1.02); ax[0].set_ylim(0, 1.02); ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[0].set_title(f"Precision-recall (each point = one score threshold)\nAP = {res_all['average_precision']:.3f}", fontsize=9)

    ax[1].plot(sweep_all["threshold"], sweep_all["f1"], "r.-", label="F1")
    ax[1].plot(sweep_all["threshold"], sweep_all["f2"], "m.-", label="F2")
    ax[1].plot(sweep_all["threshold"], sweep_all["precision"], "b.-", alpha=0.5, label="precision")
    ax[1].plot(sweep_all["threshold"], sweep_all["recall"], "g.-", alpha=0.5, label="recall")
    ax[1].set_xlabel("score threshold"); ax[1].set_ylim(0, 1.02); ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    ax[1].set_title("Metrics vs threshold (all val slices)", fontsize=9)

    ax[2].plot(sweep_all["threshold"], sweep_all["fp_on_background_slices"], "m.-", label="FP on background slices")
    ax[2].plot(sweep_all["threshold"], sweep_all["fp_on_motor_slices"], "c.-", label="FP on motor slices")
    ax[2].set_xlabel("score threshold"); ax[2].set_ylabel("false positives"); ax[2].grid(alpha=0.3); ax[2].legend(fontsize=8)
    ax[2].set_title("Where do the false positives come from?", fontsize=9)
    fig.suptitle(f"{model} / {run_dir.name}   |   hit distance = {config.HIT_DISTANCE_PX_AT_REF} px @ {config.REFERENCE_SPACING}", fontsize=10)
    plt.tight_layout()
    plt.savefig(out_dir / "pr_curve.png", dpi=120)
    plt.close()

    # ---- summary.txt ----
    L = ["=" * 78, f"EXPERIMENT : {model} / {run_dir.name}", f"SPLIT      : {args.split}", "=" * 78, ""]
    if cfg:
        p = cfg["params"]
        keys = [k for k in ("sigma", "bg_per_tomo", "epochs", "lr", "batch_size", "hm_weight", "features", "seed") if k in p]
        L += ["SETTINGS", "  " + ", ".join(f"{k}={p[k]}" for k in keys)]
        d = cfg["data"]
        L += [f"  train: {d['train_motor_slices']} motor / {d['train_background_slices']} background slices",
              f"  val  : {d['val_motor_slices']} motor / {d['val_background_slices']} background slices", ""]
    L += ["EVALUATION PROTOCOL",
          f"  hit distance : {config.HIT_DISTANCE_PX_AT_REF} px at {config.REFERENCE_SPACING} spacing (= {config.HIT_DISTANCE_NM:.1f} physical units)",
          f"  val slices   : {len(slices)} ({len(motor_ids)} motor, {n_bg} background), {n_gt} ground-truth motors", ""]
    L += ["RESULTS - all validation slices (motor + background)",
          f"  AP (area under PR curve) : {res_all['average_precision']:.4f}",
          "  best F1 : " + fmt(res_all["best_f1"]),
          "  best F2 : " + fmt(res_all["best_f2"]),
          "  fixed   : " + fmt(res_all["fixed_threshold"]),
          f"  false alarms per background slice @best F1 : {res_all['false_alarms_per_background_slice_at_best_f1']:.3f}",
          "  median localisation error @best F1         : " + (f"{res_all['median_localisation_error_px_at_best_f1']:.1f} px (original pixels)"
                                                              if res_all['median_localisation_error_px_at_best_f1'] is not None else "n/a"), ""]
    L += ["RESULTS - motor slices only (no background slices)",
          f"  AP : {res_mot['average_precision']:.4f}",
          "  best F1 : " + fmt(res_mot["best_f1"]),
          "  best F2 : " + fmt(res_mot["best_f2"]), ""]
    if train:
        L += ["TRAINING", f"  epochs={train['epochs']}  best_epoch={train['best_epoch']}  "
              f"final val loss={train['final_val_loss']:.6f}  train time={train['train_time_min']} min  params={train['model_params']:,}", ""]
    if timing:
        L += ["INFERENCE SPEED", f"  forward: {timing['forward_ms_mean']:.1f} ms/image (median {timing['forward_ms_median']:.1f})   "
              f"end-to-end: {timing['end_to_end_ms_mean']:.1f} ms/image   device: {timing['gpu'] or timing['device']}", ""]
    L += ["NOTES",
          "  * thresholds were selected on the validation set -> slightly optimistic; see 'fixed' line for an untuned value",
          "=" * 78]
    (out_dir / "summary.txt").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"saved -> {out_dir}")


if __name__ == "__main__":
    main()
