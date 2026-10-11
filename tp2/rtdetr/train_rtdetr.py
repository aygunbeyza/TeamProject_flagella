#!/usr/bin/env python
"""
RT-DETR training on the SAME data as U-Net / YOLO (processed/ultralytics_dataset).

Experiment folder  rtdetr/runs/<run_tag>/ :
  config.json, train_summary.json, results.csv, weights/{last,best}.pt
Afterwards:  predict_yolo.py --model_type rtdetr   and   evaluate.py
(evaluation uses last.pt = fixed number of epochs, same as the other models)

RT-DETR = transformer detector: it predicts a fixed set of object queries directly,
there is no NMS inside the network.
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
    ap.add_argument("--weights", default=str(config.MODEL_DIRS["rtdetr"] / "weights" / "rtdetr-l.pt"))
    ap.add_argument("--scratch", action="store_true", help="random initialisation instead of COCO-pretrained weights")
    ap.add_argument("--bg_per_tomo", type=int, default=3, choices=[0, 1, 2, 3],
                    help="motor-free slices per motor-free tomogram in training (0 = motor slices only)")
    ap.add_argument("--epochs", type=int, default=40, help="fixed number of epochs (same for all models!)")
    ap.add_argument("--imgsz", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--lr0", type=float, default=1e-4, help="AdamW start learning rate (RT-DETR default 1e-4)")
    ap.add_argument("--default_aug", action="store_true", help="use the Ultralytics default augmentation")
    ap.add_argument("--dataset_dir", default=str(config.PROCESSED_DIR / "ultralytics_dataset"))
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from ultralytics import RTDETR, settings
    settings.update({"sync": False})

    stem = Path(args.weights).stem
    run_tag = (f"{stem}{'_scratch' if args.scratch else ''}_bg{args.bg_per_tomo}_ep{args.epochs}_seed{args.seed}"
               + (f"_{args.tag}" if args.tag else ""))
    runs_dir = (config.MODEL_DIRS["rtdetr"] / "runs").resolve()
    run_dir = runs_dir / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    ds = Path(args.dataset_dir)
    data_yaml = ds / f"data_bg{args.bg_per_tomo}.yaml"
    meta = pd.read_csv(ds / "meta.csv")
    tr, va = meta[meta["split"] == "train"], meta[meta["split"] == "val"]
    data_info = {
        "train_motor_slices": int((tr["has_motor"] == 1).sum()),
        "train_background_slices": int(((tr["has_motor"] == 0) & (tr["bg_rank"] < args.bg_per_tomo)).sum()),
        "val_motor_slices": int((va["has_motor"] == 1).sum()),
        "val_background_slices": int((va["has_motor"] == 0).sum()),
    }

    device = 0 if torch.cuda.is_available() else "cpu"
    model = RTDETR(f"{stem}.yaml") if args.scratch else RTDETR(args.weights)
    n_params = int(sum(p.numel() for p in model.model.parameters()))

    if args.default_aug:
        aug = {}
    else:   # same kind of augmentation as the U-Net; no mosaic / scale / translate (keeps the pixel size)
        aug = dict(mosaic=0.0, scale=0.0, translate=0.0, degrees=15.0, flipud=0.5, fliplr=0.5,
                   hsv_h=0.0, hsv_s=0.0, hsv_v=0.2)

    cfg = {"run_tag": run_tag, "model": "rtdetr",
           "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           "params": {"weights": stem, "pretrained": not args.scratch, "imgsz": args.imgsz,
                      "batch_size": args.batch, "epochs": args.epochs, "bg_per_tomo": args.bg_per_tomo,
                      "optimizer": "AdamW", "lr0": args.lr0, "seed": args.seed, "augmentation": aug or "default"},
           "data": data_info, "model_params": n_params}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))

    t0 = time.time()
    model.train(data=str(data_yaml), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
                workers=args.workers, seed=args.seed, device=device, pretrained=not args.scratch,
                optimizer="AdamW", lr0=args.lr0,
                project=str(runs_dir), name=run_tag, exist_ok=True,
                patience=args.epochs + 100, plots=False, verbose=True, **aug)
    train_min = (time.time() - t0) / 60

    res = pd.read_csv(run_dir / "results.csv")
    res.columns = [c.strip() for c in res.columns]
    val_cols = [c for c in res.columns if c.startswith("val/") and c.endswith("_loss")]
    fit = 0.1 * res["metrics/mAP50(B)"] + 0.9 * res["metrics/mAP50-95(B)"]
    best_idx = int(fit.values.argmax())
    summary = {"run_tag": run_tag, "epochs": int(len(res)), "best_epoch": int(res["epoch"].iloc[best_idx]),
               "final_val_loss": float(res[val_cols].iloc[-1].sum()) if val_cols else None,
               "train_time_min": round(train_min, 2), "model_params": n_params,
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
