#!/usr/bin/env python
"""
YOLO / RT-DETR inference on the validation set (same interface as predict_unet.py).

Writes into the experiment folder:
  predictions_val.csv     slice_id, y, x, score     (ORIGINAL pixel coordinates)
  inference_time.json     forward_ms_mean / forward_ms_median / end_to_end_ms_mean ...
  prediction_examples.png examples with the motor NOT close to the border

Box centre -> motor position.  Score = confidence.  conf=MIN_SCORE (0.05), so the
threshold sweep in evaluate.py works exactly like for the U-Net.
Works for YOLO and RT-DETR (--model_type rtdetr).
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
from PIL import Image

import config


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run_dir", required=True, help="e.g. yolo/runs/yolov8m_bg3_ep40_seed0")
    ap.add_argument("--model_type", choices=["yolo", "rtdetr"], default="rtdetr")
    ap.add_argument("--checkpoint", choices=["last", "best"], default="last")
    ap.add_argument("--dataset_dir", default=str(config.PROCESSED_DIR / "ultralytics_dataset"))
    ap.add_argument("--imgsz", type=int, default=None, help="default: the value used for training")
    ap.add_argument("--iou", type=float, default=0.5, help="NMS IoU inside the detector")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--n_examples", type=int, default=4)
    ap.add_argument("--example_border", type=int, default=50)
    args = ap.parse_args()

    from ultralytics import YOLO, RTDETR, settings
    settings.update({"sync": False})

    run_dir = Path(args.run_dir)
    imgsz = args.imgsz
    if imgsz is None:
        cfg = json.loads((run_dir / "config.json").read_text())
        imgsz = int(cfg["params"].get("imgsz", 1024))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    Model = RTDETR if args.model_type == "rtdetr" else YOLO
    model = Model(str(run_dir / "weights" / f"{args.checkpoint}.pt"))
    print(f"loaded {args.checkpoint}.pt on {device}, imgsz={imgsz}")

    ds = Path(args.dataset_dir)
    meta = pd.read_csv(ds / "meta.csv")
    meta = meta[meta["split"] == "val"].reset_index(drop=True)
    gt = pd.read_csv(config.GT_CSV)
    gt = gt[gt["slice_id"].isin(meta["slice_id"])]

    # clear examples: motor far from the border (original coordinates)
    g = gt.merge(meta[["slice_id", "orig_h", "orig_w"]], on="slice_id")
    b = args.example_border
    ok = g[(g["y"] >= b) & (g["y"] <= g["orig_h"] - b) & (g["x"] >= b) & (g["x"] <= g["orig_w"] - b)]
    ex_ids = set(np.random.default_rng(0).permutation(ok["slice_id"].unique())[:args.n_examples])
    examples = {}

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    rows, fwd_ms, e2e_ms = [], [], []
    for i, r in enumerate(meta.itertuples()):
        img_path = str(ds / r.image) if not Path(r.image).is_absolute() else r.image
        sync()
        t0 = time.perf_counter()
        res = model.predict(img_path, imgsz=imgsz, conf=config.MIN_SCORE, iou=args.iou,
                            max_det=100, device=device, verbose=False)[0]
        sync()
        t1 = time.perf_counter()
        boxes = res.boxes
        xywh = boxes.xywh.cpu().numpy() if len(boxes) else np.zeros((0, 4))
        conf = boxes.conf.cpu().numpy() if len(boxes) else np.zeros(0)
        for (cx, cy, _, _), sc in zip(xywh, conf):
            rows.append((r.slice_id, cy / r.scale_y, cx / r.scale_x, float(sc)))   # back to ORIGINAL pixels
        if i >= args.warmup:
            fwd_ms.append(res.speed["inference"])
            e2e_ms.append((t1 - t0) * 1000)
        if r.slice_id in ex_ids:
            pts = [(yy * r.scale_y, xx * r.scale_x)
                   for yy, xx in gt.loc[gt["slice_id"] == r.slice_id, ["y", "x"]].to_numpy()]
            examples[r.slice_id] = (img_path, pts, [(cy, cx, w, h, s) for (cx, cy, w, h), s in zip(xywh, conf)])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{len(meta)} slices", flush=True)

    pred = pd.DataFrame(rows, columns=["slice_id", "y", "x", "score"])
    pred.to_csv(run_dir / "predictions_val.csv", index=False)

    timing = {"model": args.model_type, "checkpoint": args.checkpoint, "split": "val",
              "n_images_timed": len(fwd_ms), "warmup_excluded": args.warmup, "device": device,
              "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
              "forward_ms_mean": float(np.mean(fwd_ms)) if fwd_ms else None,
              "forward_ms_median": float(np.median(fwd_ms)) if fwd_ms else None,
              "end_to_end_ms_mean": float(np.mean(e2e_ms)) if e2e_ms else None,
              "note": "forward = network only (ultralytics 'inference' time); end_to_end = predict() call incl. "
                      "image read, letterbox, forward and NMS"}
    (run_dir / "inference_time.json").write_text(json.dumps(timing, indent=2))

    if examples:
        fig, ax = plt.subplots(len(examples), 2, figsize=(10, 5 * len(examples)), squeeze=False)
        for k, (sid, (path, pts, dets)) in enumerate(examples.items()):
            im = np.array(Image.open(path).convert("L"))
            ax[k, 0].imshow(im, cmap="gray")
            ax[k, 1].imshow(im, cmap="gray")
            for (yy, xx) in pts:
                for c in (0, 1):
                    ax[k, c].plot(xx, yy, "c+", ms=16, mew=2)
            for (cy, cx, w, h, s) in sorted(dets, key=lambda d: -d[4])[:3]:
                ax[k, 1].add_patch(plt.Rectangle((cx - w / 2, cy - h / 2), w, h, fill=False, ec="red", lw=1.5))
                ax[k, 1].annotate(f"{s:.2f}", (cx - w / 2, cy - h / 2), color="yellow", fontsize=8)
            ax[k, 0].set_title(f"{sid}: input (+ = ground truth)", fontsize=8)
            ax[k, 1].set_title("detections (red box, number = confidence)", fontsize=8)
            for c in range(2):
                ax[k, c].axis("off")
        fig.suptitle(run_dir.name, fontsize=10)
        plt.tight_layout()
        plt.savefig(run_dir / "prediction_examples.png", dpi=120)
        plt.close()

    print(f"{len(pred)} detections on {len(meta)} slices -> {run_dir / 'predictions_val.csv'}")
    print(json.dumps(timing, indent=2))


if __name__ == "__main__":
    main()
