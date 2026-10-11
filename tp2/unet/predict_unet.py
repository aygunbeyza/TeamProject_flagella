#!/usr/bin/env python
"""
U-Net inference on a split (default: val).

Writes, into the experiment folder:
  predictions_<split>.csv     slice_id, y, x, score      (ORIGINAL image pixel coordinates)
  inference_time.json         per-image timing (forward pass and end-to-end)
  prediction_examples.png     a few clear examples (motor not close to the image border)

predictions_<split>.csv is the ONLY interface to the evaluation: YOLO / RT-DETR
scripts just have to write the same file format to <model>/runs/<run_tag>/.

Timing protocol: the first --warmup images are excluded, CUDA is synchronised
before / after every measurement, image loading is NOT part of "forward_ms";
"end_to_end_ms" = preprocessing (normalise + rescale + pad) + forward + peak decoding.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config
from common.imaging import load_normalized, rescale_to_reference
from common.peaks import detect_peaks
from unet_model import UNet, pad_to_multiple


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="e.g. unet/runs/sigma12_bg3_ep40_seed0")
    ap.add_argument("--checkpoint", choices=["last", "best"], default="last",
                    help="'last' = weights after the final epoch (fair fixed-epoch comparison)")
    ap.add_argument("--split", default="val")
    ap.add_argument("--min_distance", type=int, default=20, help="NMS radius in pixels")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n_examples", type=int, default=4)
    ap.add_argument("--example_border", type=int, default=50, help="examples: motor at least this far from the border (px)")
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(run_dir / f"model_{args.checkpoint}.pth", map_location=device)
    model = UNet(ckpt["features"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"loaded {args.checkpoint} checkpoint (epoch {ckpt['epoch']}) on {device}")

    slices = pd.read_csv(config.SLICES_CSV)
    slices = slices[slices["split"] == args.split].reset_index(drop=True)
    gt = pd.read_csv(config.GT_CSV)
    gt = gt[gt["slice_id"].isin(slices["slice_id"])]

    # clear examples: motor slices whose motor is far from the image border
    g = gt.merge(slices[["slice_id", "height", "width"]], on="slice_id")
    b = args.example_border
    ok = g[(g["y"] >= b) & (g["y"] <= g["height"] - b) & (g["x"] >= b) & (g["x"] <= g["width"] - b)]
    ex_ids = list(np.random.default_rng(0).permutation(ok["slice_id"].unique())[:args.n_examples])
    examples = {}

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    rows, fwd_ms, e2e_ms = [], [], []
    for i, r in enumerate(slices.itertuples()):
        img = load_normalized(r.path)                      # file loading is not part of the timing
        t_loaded = time.perf_counter()
        img_r, sy, sx = rescale_to_reference(img, r.voxel_spacing, config.REFERENCE_SPACING)
        x, (h, w) = pad_to_multiple(torch.from_numpy(np.ascontiguousarray(img_r))[None, None])
        x = x.to(device)
        sync()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(x)
        sync()
        t1 = time.perf_counter()
        hm = out[0, 0, :h, :w].cpu().numpy()
        peaks = detect_peaks(hm, config.MIN_SCORE, args.min_distance)
        t2 = time.perf_counter()

        for (py, px, sc) in peaks:
            rows.append((r.slice_id, py / sy, px / sx, sc))      # back to ORIGINAL pixel coordinates
        if i >= args.warmup:
            fwd_ms.append((t1 - t0) * 1000)
            e2e_ms.append((t2 - t_loaded) * 1000)
        if r.slice_id in ex_ids:
            pts = [(yy * sy, xx * sx) for yy, xx in gt.loc[gt["slice_id"] == r.slice_id, ["y", "x"]].to_numpy()]
            examples[r.slice_id] = (img_r, hm, pts, peaks)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(slices)} slices", flush=True)

    pred = pd.DataFrame(rows, columns=["slice_id", "y", "x", "score"])
    pred.to_csv(run_dir / f"predictions_{args.split}.csv", index=False)

    timing = {"model": "unet", "checkpoint": args.checkpoint, "split": args.split,
              "n_images_timed": len(fwd_ms), "warmup_excluded": args.warmup, "device": device,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "forward_ms_mean": float(np.mean(fwd_ms)) if fwd_ms else None,
              "forward_ms_median": float(np.median(fwd_ms)) if fwd_ms else None,
              "end_to_end_ms_mean": float(np.mean(e2e_ms)) if e2e_ms else None,
              "note": "forward = network only; end_to_end = rescale + pad + forward + peak decoding (image file loading excluded)"}
    (run_dir / "inference_time.json").write_text(json.dumps(timing, indent=2))

    # ---- example figure ----
    if examples:
        fig, ax = plt.subplots(len(examples), 3, figsize=(12, 4 * len(examples)), squeeze=False)
        for k, (sid, (img_r, hm, pts, peaks)) in enumerate(examples.items()):
            ax[k, 0].imshow(img_r, cmap="gray")
            ax[k, 1].imshow(hm, cmap="hot", vmin=0, vmax=1)
            ax[k, 2].imshow(img_r, cmap="gray")
            for (yy, xx) in pts:
                for c in (0, 1, 2):
                    ax[k, c].plot(xx, yy, "c+", ms=16, mew=2)
            for (py, px, sc) in peaks[:3]:
                ax[k, 2].plot(px, py, "rx", ms=11, mew=2)
                ax[k, 2].annotate(f"{sc:.2f}", (px, py), color="yellow", fontsize=8)
            ax[k, 0].set_title(f"{sid}: input (+ = ground truth)", fontsize=8)
            ax[k, 1].set_title("predicted heat map", fontsize=8)
            ax[k, 2].set_title("detections (x = prediction, number = score)", fontsize=8)
            for c in range(3):
                ax[k, c].axis("off")
        fig.suptitle(run_dir.name, fontsize=10)
        plt.tight_layout()
        plt.savefig(run_dir / "prediction_examples.png", dpi=120)
        plt.close()

    print(f"{len(pred)} detections on {len(slices)} slices -> {run_dir / f'predictions_{args.split}.csv'}")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
