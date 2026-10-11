#!/usr/bin/env python
"""
DATA PREPARATION (shared by U-Net, YOLO and RT-DETR; run ONCE).
Usage: python prepare_data.py

Builds the single source of truth that U-Net, YOLO and RT-DETR all use:

  processed/slices.csv   one row per image:
                         slice_id, split, tomo_id, path, has_motor, voxel_spacing,
                         height, width, bg_rank
  processed/gt.csv       one row per ground-truth motor: slice_id, y, x (original pixels)
  processed/dataset_stats.json

Design decisions
  * The train/val split of the tomograms is read from train_ids.txt / val_ids.txt
    (so it stays identical to your earlier experiments); if they do not exist a
    seeded 80/20 split is created and written there.
  * "motor slice"      = the slice containing an annotated motor.
    "background slice" = a slice from a tomogram without any motor.
  * Background slices are chosen ONCE here with a fixed seed (middle 30-70 % of the
    tomogram), for train AND val.  Hence the validation set is identical for every
    model and every run.  `bg_rank` (0,1,2) orders the background slices of a tomogram,
    so training with fewer background slices (e.g. 0 = motor slices only) just selects
    bg_rank < k from the SAME file.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[0]))

import numpy as np
import pandas as pd
from PIL import Image

import config


def read_ids(path):
    with open(path) as f:
        return [line.strip() for line in f if line.strip()]


def get_split(df, seed, val_fraction):
    if config.TRAIN_IDS_TXT.exists() and config.VAL_IDS_TXT.exists():
        train_ids, val_ids = read_ids(config.TRAIN_IDS_TXT), read_ids(config.VAL_IDS_TXT)
        print(f"split read from {config.TRAIN_IDS_TXT.name} / {config.VAL_IDS_TXT.name}")
    else:
        rng = np.random.default_rng(seed)
        per_tomo = df.groupby("tomo_id")["Number of motors"].max()
        train_ids, val_ids = [], []
        for has in (True, False):                       # stratify by motor / no motor
            ids = np.array(sorted(per_tomo[(per_tomo > 0) == has].index))
            rng.shuffle(ids)
            n_val = int(round(val_fraction * len(ids)))
            val_ids += list(ids[:n_val])
            train_ids += list(ids[n_val:])
        config.TRAIN_IDS_TXT.write_text("\n".join(sorted(train_ids)) + "\n")
        config.VAL_IDS_TXT.write_text("\n".join(sorted(val_ids)) + "\n")
        print("no split files found -> created a seeded stratified split")
    overlap = set(train_ids) & set(val_ids)
    if overlap:
        raise RuntimeError(f"{len(overlap)} tomograms are in both train and val")
    return set(train_ids), set(val_ids)


def image_size(path):
    with Image.open(path) as im:
        return im.size[1], im.size[0]          # (height, width)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bg_per_tomo", type=int, default=config.BG_SLICES_PER_TOMO)
    ap.add_argument("--seed", type=int, default=config.SEED)
    args = ap.parse_args()

    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(config.LABELS_CSV)
    print(f"{len(df)} label rows, {df['tomo_id'].nunique()} tomograms")
    train_ids, val_ids = get_split(df, config.SPLIT_SEED, config.VAL_FRACTION)

    rng = np.random.default_rng(args.seed)
    slice_rows, gt_rows = [], []
    n_missing = 0

    # ---- motor slices (a slice may contain more than one motor) ----
    motor_df = df[df["Number of motors"] > 0]
    for (tomo, z), g in motor_df.groupby(["tomo_id", "Motor axis 0"]):
        split = "train" if tomo in train_ids else "val" if tomo in val_ids else None
        if split is None:
            continue
        path = config.IMAGE_DIR / tomo / f"slice_{int(z):04d}.jpg"
        if not path.exists():
            n_missing += 1
            continue
        h, w = image_size(path)
        sid = f"{tomo}_{int(z):04d}"
        slice_rows.append(dict(slice_id=sid, split=split, tomo_id=tomo, path=str(path), has_motor=1,
                               voxel_spacing=float(g["Voxel spacing"].iloc[0]),
                               height=h, width=w, bg_rank=-1))
        for _, r in g.iterrows():
            gt_rows.append(dict(slice_id=sid, y=float(r["Motor axis 1"]), x=float(r["Motor axis 2"])))

    # ---- background slices (tomograms without a motor) ----
    per_tomo_max = df.groupby("tomo_id")["Number of motors"].max()
    spacing = df.drop_duplicates("tomo_id").set_index("tomo_id")["Voxel spacing"].to_dict()
    for tomo in sorted(per_tomo_max[per_tomo_max == 0].index):
        split = "train" if tomo in train_ids else "val" if tomo in val_ids else None
        d = config.IMAGE_DIR / tomo
        if split is None or not d.is_dir():
            continue
        files = sorted(p.name for p in d.iterdir() if p.suffix == ".jpg")
        if not files:
            continue
        lo, hi = int(config.BG_REGION[0] * len(files)), int(config.BG_REGION[1] * len(files))
        cand = files[lo:hi] or files
        pick = rng.choice(cand, size=min(args.bg_per_tomo, len(cand)), replace=False)
        for rank, fn in enumerate(pick):
            h, w = image_size(d / fn)
            slice_rows.append(dict(slice_id=f"{tomo}_{Path(fn).stem.split('_')[-1]}", split=split,
                                   tomo_id=tomo, path=str(d / fn), has_motor=0,
                                   voxel_spacing=float(spacing[tomo]), height=h, width=w, bg_rank=rank))

    slices = pd.DataFrame(slice_rows).sort_values(["split", "slice_id"]).reset_index(drop=True)
    gt = pd.DataFrame(gt_rows)
    slices.to_csv(config.SLICES_CSV, index=False)
    gt.to_csv(config.GT_CSV, index=False)

    stats = {"n_missing_motor_slices": n_missing, "bg_per_tomo": args.bg_per_tomo, "seed": args.seed,
             "reference_spacing": config.REFERENCE_SPACING,
             "voxel_spacing_min": float(slices["voxel_spacing"].min()),
             "voxel_spacing_max": float(slices["voxel_spacing"].max())}
    for split in ("train", "val"):
        s = slices[slices["split"] == split]
        stats[split] = {"tomograms": int(s["tomo_id"].nunique()),
                        "motor_slices": int((s["has_motor"] == 1).sum()),
                        "background_slices": int((s["has_motor"] == 0).sum()),
                        "gt_motors": int(gt["slice_id"].isin(s["slice_id"]).sum())}
    (config.PROCESSED_DIR / "dataset_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"wrote {config.SLICES_CSV} and {config.GT_CSV}")


if __name__ == "__main__":
    main()
