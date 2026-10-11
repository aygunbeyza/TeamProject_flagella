#!/usr/bin/env python
"""
RT-DETR: automatic parameter search (independent of YOLO: own datasets, own runs, own results).

Stage 1  learning rate  lr0   (box 48 px, imgsz 1024)          -> best lr0
Stage 2  label box size       (best lr0, imgsz 1024)            -> best box
Stage 3  input size imgsz     (best lr0, best box)              -> best imgsz
Every run = train (fixed number of epochs) -> predict (last.pt) -> evaluate (same val set / same metric).
Selection criterion: best F1 on the validation set (ties: AP).  Slightly optimistic (val is used twice).

Writes  rtdetr/results/search<_scratch>.csv / .md   (table of all runs, best marked)
        rtdetr/results/best<_scratch>.json
Re-runnable: finished runs (eval_val/metrics.json exists) are skipped.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

import config

ROOT = config.PROJECT_DIR
RUNS = ROOT / "rtdetr" / "runs"
RESULTS = ROOT / "rtdetr" / "results"
PY = sys.executable


def md_table(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join("" if pd.isna(v) else str(v) for v in r.values) + " |")
    return "\n".join(lines)


def sh(cmd):
    print("\n$ " + " ".join(str(c) for c in cmd), flush=True)
    subprocess.run([str(c) for c in cmd], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "rtdetr" / "weights" / "rtdetr-l.pt"))
    ap.add_argument("--scratch", action="store_true")
    ap.add_argument("--lrs", nargs="+", default=["5e-5", "1e-4", "2e-4", "4e-4"])
    ap.add_argument("--boxes", nargs="+", type=int, default=[32, 48, 64])
    ap.add_argument("--imgszs", nargs="+", type=int, default=[640, 1024])
    ap.add_argument("--box0", type=int, default=48, help="box size used in stage 1")
    ap.add_argument("--imgsz0", type=int, default=1024, help="input size used in stages 1 and 2")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bg_per_tomo", type=int, default=3)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=config.SEED)
    ap.add_argument("--dry_run", action="store_true", help="only print the plan of stage 1")
    a = ap.parse_args()

    stem = Path(a.weights).stem
    sfx = "_scratch" if a.scratch else ""
    RESULTS.mkdir(parents=True, exist_ok=True)
    table = {}          # run_name -> row

    def name_of(lr, box, img):
        return f"{stem}{sfx}_bg{a.bg_per_tomo}_ep{a.epochs}_seed{a.seed}_lr{lr}_box{box}_img{img}"

    def dataset(box):
        ds = ROOT / "processed" / f"rtdetr_dataset_box{box}"
        if not (ds / "val.txt").exists():
            sh([PY, "common/export_yolo_dataset.py", "--box_px", box, "--out_dir", ds])
        return ds

    def read_row(run_dir, lr, box, img):
        m = json.loads((run_dir / "eval_val" / "metrics.json").read_text())
        al, tr = m["all_slices"], (m.get("train") or {})
        return {"run": run_dir.name, "lr0": lr, "box_px": box, "imgsz": img,
                "F1": al["best_f1"]["f1"], "F1_thr": al["best_f1"]["threshold"],
                "P@F1": al["best_f1"]["precision"], "R@F1": al["best_f1"]["recall"],
                "AP": al["average_precision"], "F1@0.5": al["fixed_threshold"]["f1"],
                "FP_bg@F1": al["best_f1"]["fp_on_background_slices"],
                "best_epoch": tr.get("best_epoch"), "train_min": tr.get("train_time_min")}

    def run_one(lr, box, img):
        name = name_of(lr, box, img)
        run_dir = RUNS / name
        if not (run_dir / "eval_val" / "metrics.json").exists():
            print(f"\n=========== RT-DETR  lr0={lr}  box={box}px  imgsz={img}  -> {name}", flush=True)
            try:
                ds = dataset(box)
                sh([PY, "rtdetr/train_rtdetr.py", "--weights", a.weights, "--bg_per_tomo", a.bg_per_tomo,
                    "--epochs", a.epochs, "--batch", a.batch, "--imgsz", img, "--lr0", lr,
                    "--dataset_dir", ds, "--seed", a.seed, "--tag", f"lr{lr}_box{box}_img{img}"]
                   + (["--scratch"] if a.scratch else []))
                sh([PY, "rtdetr/predict_rtdetr.py", "--run_dir", run_dir, "--checkpoint", "last", "--dataset_dir", ds])
                sh([PY, "evaluate.py", "--run_dir", run_dir])
            except subprocess.CalledProcessError as e:
                print(f"!!! FAILED {name}: {e}", flush=True)
                return None
        else:
            print(f"[skip] {name} already evaluated")
        row = read_row(run_dir, lr, box, img)
        table[name] = row
        return row

    def pick(rows):
        rows = [r for r in rows if r is not None]
        if not rows:
            sys.exit("no successful run in this stage -> stop")
        return max(rows, key=lambda r: (r["F1"], r["AP"]))

    def write_tables(best=None):
        df = pd.DataFrame(list(table.values()))
        if df.empty:
            return df
        df["selected"] = ""
        if best is not None:
            df.loc[df["run"] == best["run"], "selected"] = "<- BEST"
        df = df.sort_values(["lr0", "box_px", "imgsz"]).reset_index(drop=True)
        df.round(4).to_csv(RESULTS / f"search{sfx}.csv", index=False)
        md = ["# RT-DETR parameter search" + (" (from scratch)" if a.scratch else " (COCO-pretrained)"), "",
              f"Fixed: {a.epochs} epochs, {a.bg_per_tomo} motor-free slices per tomogram, batch {a.batch}, "
              f"same validation set and metric as all models. Selection: best F1 on val (ties: AP).", "",
              md_table(df.round(3))]
        (RESULTS / f"search{sfx}.md").write_text("\n".join(md) + "\n")
        return df

    if a.dry_run:
        print("stage 1 (lr0):", [name_of(lr, a.box0, a.imgsz0) for lr in a.lrs])
        print("stage 2 (box) : lr0 = best of stage 1, boxes", a.boxes)
        print("stage 3 (imgsz): lr0, box = best so far, imgsz", a.imgszs)
        return

    s1 = [run_one(lr, a.box0, a.imgsz0) for lr in a.lrs]
    b1 = pick(s1); write_tables(b1)
    print(f"\n>>> stage 1 best lr0 = {b1['lr0']}  (F1 {b1['F1']:.3f})", flush=True)

    s2 = [run_one(b1["lr0"], box, a.imgsz0) for box in a.boxes]
    b2 = pick(s2 + [b1]); write_tables(b2)
    print(f"\n>>> stage 2 best box = {b2['box_px']} px  (F1 {b2['F1']:.3f})", flush=True)

    s3 = [run_one(b2["lr0"], b2["box_px"], img) for img in a.imgszs]
    best = pick(s3 + [b2]); df = write_tables(best)
    print(f"\n>>> stage 3 best imgsz = {best['imgsz']}  (F1 {best['F1']:.3f})", flush=True)

    (RESULTS / f"best{sfx}.json").write_text(json.dumps(
        {"run": best["run"], "run_dir": str(RUNS / best["run"]), "lr0": best["lr0"], "box_px": best["box_px"],
         "imgsz": best["imgsz"], "F1": best["F1"], "AP": best["AP"]}, indent=2))
    print("\n" + df.round(3).to_string(index=False))
    print(f"\nBEST RT-DETR run: {best['run']}\ntable -> {RESULTS / ('search' + sfx + '.md')}")


if __name__ == "__main__":
    main()
