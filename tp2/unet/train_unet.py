#!/usr/bin/env python
"""
Train the U-Net heat-map detector.   (training ONLY; no evaluation here)

Experiment folder (one per experiment, everything needed to reproduce it):
  unet/runs/<run_tag>/config.json          all hyper-parameters + data counts
                      train_log.csv        per-epoch train / val loss
                      loss_curve.png
                      model_last.pth       weights after the LAST epoch   (used for evaluation)
                      model_best.pth       weights with the lowest val loss
                      train_summary.json   epochs, best epoch, train time, #params

Fair-comparison rules implemented here
  * fixed number of epochs, NO early stopping (--epochs; use the same for YOLO / RT-DETR)
  * augmentation only on the training set, the validation set is never altered
  * --bg_per_tomo 3  -> trained with motor slices + background slices
    --bg_per_tomo 0  -> trained on motor slices only  (the 'positive-only' comparison model)
    The validation set is the same for both.
"""
import argparse
import csv
import datetime
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from unet_dataset import MotorSliceDataset, pad_batch
from unet_model import UNet, weighted_mse, PAD_MULTIPLE


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sigma", type=float, default=12.0, help="Gaussian sigma of the target heat map (pixels at reference spacing)")
    ap.add_argument("--bg_per_tomo", type=int, default=config.BG_SLICES_PER_TOMO,
                    help="background slices per motor-free tomogram used for training (0 = motor slices only)")
    ap.add_argument("--epochs", type=int, default=40, help="fixed number of epochs (same for all models!)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--hm_weight", type=float, default=10.0, help="extra loss weight near the motor")
    ap.add_argument("--features", type=int, nargs="+", default=[32, 64, 128, 256])
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--no_intensity_aug", action="store_true", help="disable contrast/brightness and noise augmentation")
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--tag", type=str, default="", help="optional suffix for the run folder name")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    run_tag = f"sigma{int(args.sigma)}_bg{args.bg_per_tomo}_ep{args.epochs}_seed{args.seed}" + (f"_{args.tag}" if args.tag else "")
    run_dir = config.MODEL_DIRS["unet"] / "runs" / run_tag
    run_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------ data
    slices = pd.read_csv(config.SLICES_CSV)
    gt = pd.read_csv(config.GT_CSV)
    train = slices[(slices["split"] == "train") &
                   ((slices["has_motor"] == 1) | (slices["bg_rank"] < args.bg_per_tomo))]
    val = slices[slices["split"] == "val"]                 # always the full, fixed validation set
    n_tr_fg, n_tr_bg = int((train["has_motor"] == 1).sum()), int((train["has_motor"] == 0).sum())
    n_va_fg, n_va_bg = int((val["has_motor"] == 1).sum()), int((val["has_motor"] == 0).sum())

    aug_cfg = {"p_intensity": 0.0, "p_noise": 0.0} if args.no_intensity_aug else None
    train_ds = MotorSliceDataset(train, gt, args.sigma, config.REFERENCE_SPACING, augment=True, aug_cfg=aug_cfg, seed=args.seed)
    val_ds = MotorSliceDataset(val, gt, args.sigma, config.REFERENCE_SPACING, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, collate_fn=pad_batch)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, collate_fn=pad_batch)

    model = UNet(args.features).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    n_params = sum(p.numel() for p in model.parameters())

    cfg = {"run_tag": run_tag, "model": "unet", "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
           "params": {**vars(args), "pad_multiple": PAD_MULTIPLE, "reference_spacing": config.REFERENCE_SPACING,
                      "early_stopping": False, "device": device},
           "data": {"train_motor_slices": n_tr_fg, "train_background_slices": n_tr_bg,
                    "val_motor_slices": n_va_fg, "val_background_slices": n_va_bg},
           "model_params": n_params}
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=2))
    print(json.dumps(cfg, indent=2))

    # ------------------------------------------------------------ training loop
    log_path = run_dir / "train_log.csv"
    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "val_loss", "seconds"])

    train_losses, val_losses = [], []
    best_val, best_ep = float("inf"), 0
    t0 = datetime.datetime.now()
    for ep in range(1, args.epochs + 1):
        t_ep = datetime.datetime.now()
        model.train()
        tot = 0.0
        for img, hm, _ in train_loader:
            img, hm = img.to(device), hm.to(device)
            opt.zero_grad()
            loss = weighted_mse(model(img), hm, args.hm_weight)
            loss.backward()
            opt.step()
            tot += loss.item()
        tr = tot / len(train_loader)

        model.eval()
        tot = 0.0
        with torch.no_grad():
            for img, hm, _ in val_loader:
                img, hm = img.to(device), hm.to(device)
                tot += weighted_mse(model(img), hm, args.hm_weight).item()
        va = tot / len(val_loader)

        train_losses.append(tr)
        val_losses.append(va)
        secs = (datetime.datetime.now() - t_ep).total_seconds()
        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow([ep, f"{tr:.8f}", f"{va:.8f}", f"{secs:.1f}"])

        ckpt = {"state_dict": model.state_dict(), "features": args.features, "sigma": args.sigma, "epoch": ep}
        if va < best_val:
            best_val, best_ep = va, ep
            torch.save(ckpt, run_dir / "model_best.pth")
        torch.save(ckpt, run_dir / "model_last.pth")      # overwritten every epoch -> final weights
        print(f"epoch {ep:03d}/{args.epochs} | train {tr:.6f} | val {va:.6f} | {secs:.0f}s"
              + ("  <- best" if ep == best_ep else ""), flush=True)

    minutes = (datetime.datetime.now() - t0).total_seconds() / 60

    # ------------------------------------------------------------ outputs
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, args.epochs + 1), train_losses, label="train")
    plt.plot(range(1, args.epochs + 1), val_losses, label="val")
    plt.axvline(best_ep, ls="--", c="gray", lw=1, label=f"lowest val loss (epoch {best_ep})")
    plt.xlabel("epoch")
    plt.ylabel(f"weighted MSE (hm_weight={args.hm_weight:g})")
    plt.title(run_tag, fontsize=9)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(run_dir / "loss_curve.png", dpi=120)
    plt.close()

    summary = {"run_tag": run_tag, "epochs": args.epochs, "best_epoch": best_ep,
               "best_val_loss": best_val, "final_val_loss": val_losses[-1], "final_train_loss": train_losses[-1],
               "train_time_min": round(minutes, 1), "model_params": n_params, "device": device,
               "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None}
    (run_dir / "train_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"saved -> {run_dir}")


if __name__ == "__main__":
    main()
