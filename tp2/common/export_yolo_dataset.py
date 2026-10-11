#!/usr/bin/env python
"""
Export processed/slices.csv + gt.csv to the Ultralytics detection format (used by YOLO AND RT-DETR).
Usage: python common/export_yolo_dataset.py [--box_px 48]

Images : normalised (0.5/99.5 percentiles) and rescaled to the reference voxel spacing - exactly the same
         pipeline as for the U-Net (common/imaging.py) - saved as 8-bit grey PNG.
Labels : a motor is a point, YOLO needs a box -> fixed square box of --box_px pixels (at reference
         spacing) centred on the motor. Near the image border the box is shrunk SYMMETRICALLY so that
         the box centre (the only thing the evaluation uses) stays exactly on the motor.
         Slices without a motor get an empty label file (= background image).

Output (default: processed/ultralytics_dataset/):
  images/{train,val}/<slice_id>.png        labels/{train,val}/<slice_id>.txt
  train_bg0.txt ... train_bg3.txt          image lists for training (bg k = motor slices + bg_rank < k)
  val.txt                                  the fixed validation set (always all 255 slices)
  data_bg0.yaml ... data_bg3.yaml          dataset configs
  meta.csv                                 per image: original / rescaled size and scale (to map predictions back)
  dataset_info.json
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from PIL import Image

import config
from common.imaging import load_normalized, rescale_to_reference


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out_dir", default=str(config.PROCESSED_DIR / "ultralytics_dataset"))
    ap.add_argument("--box_px", type=float, default=48.0, help="box side in pixels at the reference spacing")
    ap.add_argument("--redo", action="store_true", help="rewrite images that already exist")
    args = ap.parse_args()

    out = Path(args.out_dir).resolve()
    slices = pd.read_csv(config.SLICES_CSV)
    gt = pd.read_csv(config.GT_CSV)
    gt_by = {sid: g[["y", "x"]].to_numpy(float) for sid, g in gt.groupby("slice_id")}
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    meta, n_boxes = [], 0
    for i, r in enumerate(slices.itertuples()):
        img_p = out / "images" / r.split / f"{r.slice_id}.png"
        lab_p = out / "labels" / r.split / f"{r.slice_id}.txt"
        img = load_normalized(r.path)
        img_r, sy, sx = rescale_to_reference(img, r.voxel_spacing, config.REFERENCE_SPACING)
        H, W = img_r.shape
        if args.redo or not img_p.exists():
            Image.fromarray((img_r * 255.0 + 0.5).astype(np.uint8), mode="L").save(img_p)

        lines = []
        for (y, x) in gt_by.get(r.slice_id, []):
            cy, cx = y * sy, x * sx                                   # motor in the rescaled image
            half = max(2.0, min(args.box_px / 2, cx, W - cx, cy, H - cy))
            lines.append(f"0 {cx / W:.6f} {cy / H:.6f} {2 * half / W:.6f} {2 * half / H:.6f}")
        lab_p.write_text("\n".join(lines) + ("\n" if lines else ""))
        n_boxes += len(lines)
        meta.append(dict(slice_id=r.slice_id, split=r.split, has_motor=r.has_motor, bg_rank=r.bg_rank,
                         orig_h=r.height, orig_w=r.width, new_h=H, new_w=W, scale_y=H / r.height,
                         scale_x=W / r.width, image=str(img_p)))
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(slices)} images", flush=True)

    meta = pd.DataFrame(meta)
    meta.to_csv(out / "meta.csv", index=False)

    # image lists: training with k motor-free slices per tomogram = motor slices + bg_rank < k
    tr, va = meta[meta["split"] == "train"], meta[meta["split"] == "val"]
    (out / "val.txt").write_text("\n".join(va["image"]) + "\n")
    for k in range(int(tr["bg_rank"].max()) + 2):
        sel = tr[(tr["has_motor"] == 1) | (tr["bg_rank"] < k)]
        (out / f"train_bg{k}.txt").write_text("\n".join(sel["image"]) + "\n")
        (out / f"data_bg{k}.yaml").write_text(
            f"path: {out}\ntrain: {out / f'train_bg{k}.txt'}\nval: {out / 'val.txt'}\nnames:\n  0: motor\n")

    info = {"box_px_at_reference": args.box_px, "reference_spacing": config.REFERENCE_SPACING,
            "n_images": len(meta), "n_boxes": n_boxes,
            "train_images": len(tr), "val_images": len(va),
            "train_images_per_bg": {f"bg{k}": int(((tr['has_motor'] == 1) | (tr['bg_rank'] < k)).sum())
                                    for k in range(int(tr["bg_rank"].max()) + 2)}}
    (out / "dataset_info.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
