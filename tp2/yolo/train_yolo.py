#!/usr/bin/env python
"""
Train YOLO (Ultralytics) on the shared dataset.   (training ONLY; no evaluation here)
Usage: python yolo/train_yolo.py --bg_per_tomo 3 --epochs 40

Prerequisite: python common/export_yolo_dataset.py   (-> processed/ultralytics_dataset/)

Experiment folder  yolo/runs/<run_tag>/ :
  config.json, train_summary.json     same keys as the U-Net (needed by evaluate.py)
  weights/last.pt, weights/best.pt    last = used for evaluation (fixed epochs, no early stopping)
  results.csv                         per-epoch losses / metrics written by Ultralytics

Fair-comparison rules: same epochs as the U-Net, no early stopping, evaluate the LAST weights,
same images / normalisation / voxel-spacing rescaling (they were baked into the exported PNGs).
Augmentation is matched to the U-Net (flip, rotation +-15 deg, brightness); no scale / mosaic /
translate because the pixel size is already normalised (use --default_aug for Ultralytics defaults).
"""
import argparse
import datetime
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch

import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(config.MODEL_DIRS["yolo"] / "weights" / "yolov8m.pt"))
    ap.add_argument("--scratch", action="store_true", help="random initialisation instead of COCO-pretrained weights")
    ap.add_argument("--bg_per_tomo", type=int, default=3, choices=[0, 1, 2, 3],
                    help="motor-free slices per tomogram used for training (0 = motor slices only)")
    ap.add_argument("--epochs", type=int, default=40, help="fixed number of epochs (same for all models!)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--default_aug", action="store_true", help="use the Ultralytics default augmentation")
    ap.add_argument("--dataset_dir", default=str(config.PROCESSED_DIR / "ultralytics_dataset"))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from ultralytics import YOLO, settings
    settings.update({"sync": False})                   # no network access needed

    stem = Path(args.weights).stem
    run_tag = (f"{stem}{'_scratch' if args.scratch else ''}_bg{args.bg_per_tomo}_ep{args.epochs}_seed{args.seed}"
               + (f"_{args.tag}" if args.tag else ""))
    runs_dir = (config.MODEL_DIRS["yolo"] / "runs").resolve()
    run_dir = runs_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    ds = Path(args.dataset_dir)
    data_yaml = ds / f"data_bg{args.bg_per_tomo}.yaml"
    device = 0 if torch.cuda.is_available() else "cpu"

    # ---------------------------------------------------------- bookkeeping (same keys as the U-Net)
    meta = pd.read_csv(ds / "meta.csv")
    tr = meta[(meta["split"] == "train") & ((meta["has_motor"] == 1) | (meta["bg_rank"] < args.bg_per_tomo))]
    va = meta[meta["split"] == "val"]
    box_px = json.loads((ds / "dataset_info.json").read_text())["box_px_at_reference"]
    model = YOLO(f"{stem}.yaml") if args.scratch else YOLO(args.weights)
    n_params = sum(p.numel() for p in model.model.parameters())
    cfg = {"run_tag": run_tag, "model": "yolo", "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           "params": {"weights": stem, "pretrained": not args.scratch, "imgsz": args.imgsz, "batch_size": args.batch,
                      "epochs": args.epochs, "bg_per_tomo": args.bg_per_tomo, "seed": args.seed,
                      "box_px_at_reference": box_px, "reference_spacing": config.REFERENCE_SPACING,
                      "augmentation": "ultralytics default" if args.default_aug else "matched to U-Net",
                      "early_stopping": False, "device": str(device)},
           "data": {"train_motor_slices": int((tr["has_motor"] == 1).sum()),
                    "train_background_slices": int((tr["has_motor"] == 0).sum()),
                    "val_motor_slices": int((va["has_motor"] == 1).sum()),
                    "val_background_slices": int((va["has_motor"] == 0).sum())},
           "model_params": n_params}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2), flush=True)

    # ---------------------------------------------------------- training
    aug = {} if args.default_aug else dict(mosaic=0.0, scale=0.0, translate=0.0, degrees=15.0, flipud=0.5,
                                           fliplr=0.5, hsv_h=0.0, hsv_s=0.0, hsv_v=0.2)
    t0 = time.time()
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, seed=args.seed, device=device, pretrained=not args.scratch,
                project=str(runs_dir), name=run_tag, exist_ok=True,
                patience=args.epochs + 100,            # early stopping effectively off
                plots=False, verbose=True, **aug)
    minutes = (time.time() - t0) / 60

    # ---------------------------------------------------------- summary (same keys as the U-Net)
    res = pd.read_csv(run_dir / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    val_cols = [c for c in res.columns if c.startswith("val/") and c.endswith("_loss")]
    val_loss = res[val_cols].sum(axis=1)                   # box + cls + dfl
    fitness = 0.1 * res["metrics/mAP50(B)"] + 0.9 * res["metrics/mAP50-95(B)"]
    best_idx = int(fitness.idxmax())
    summary = {"run_tag": run_tag, "epochs": int(len(res)), "best_epoch": int(res["epoch"].iloc[best_idx]),
               "best_val_loss": float(val_loss.min()), "final_val_loss": float(val_loss.iloc[-1]),
               "train_time_min": round(minutes, 1), "model_params": n_params, "device": str(device),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "note": "val loss = box + cls + dfl summed; best epoch = highest Ultralytics fitness (not used for evaluation)"}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
