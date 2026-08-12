import torch.nn.functional as F
import os, json, csv
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from PIL import Image

from train_model import UNet, DEVICE, RUN_TAG, RUN_DIR, OUT_ROOT, BASE_DIR, HIT_DIST

MODEL_PATH   = os.path.join(RUN_DIR, "unet_model.pth")
PROJ_DIR     = "/data/horse/ws/beay097h-teamproject/TeamProject_flagella"
NEG_SLICES   = 3          # kac negatif slice degerlendirilecek (tomogram basina)
MIN_DISTANCE = 20


def detect_peaks(heatmap, threshold=0.3, min_distance=MIN_DISTANCE):
    local_max = maximum_filter(heatmap, size=min_distance)
    peaks_mask = (heatmap == local_max) & (heatmap >= threshold)
    ys, xs = np.where(peaks_mask)
    scores = heatmap[ys, xs]
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]
    keep = []
    for i in range(len(ys)):
        if all(np.hypot(ys[i] - ys[j], xs[i] - xs[j]) >= min_distance for j in keep):
            keep.append(i)
    return [(ys[i], xs[i], scores[i]) for i in keep]


def evaluate_detections(all_detections, all_ground_truths, hit_distance=HIT_DIST):
    tp = fp = fn = 0
    for dets, gt_list in zip(all_detections, all_ground_truths):
        gt_matched = [False] * len(gt_list)
        for (dy, dx, score) in dets:
            matched = False
            for i, gt in enumerate(gt_list):
                if gt_matched[i]:
                    continue
                if np.hypot(dy - gt[0], dx - gt[1]) <= hit_distance:
                    tp += 1; gt_matched[i] = True; matched = True; break
            if not matched:
                fp += 1
        fn += gt_matched.count(False)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall    = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1, tp, fp, fn


def load_norm(path):
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
    return np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


if __name__ == "__main__":
    print("=" * 70)
    print(f"EVALUATING RUN : {RUN_TAG}")
    print(f"MODEL          : {MODEL_PATH}")
    print(f"HIT DISTANCE   : {HIT_DIST} px")
    print("=" * 70)

    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    with open(os.path.join(BASE_DIR, "val_ids.txt")) as f:
        val_ids = set(line.strip() for line in f if line.strip())
    df = pd.read_csv(os.path.join(BASE_DIR, "train_labels.csv"))
    val_df = df[df["tomo_id"].isin(val_ids)]

    pos_df = val_df[val_df["Number of motors"] > 0]
    neg_ids = sorted(val_df[val_df["Number of motors"] == 0]["tomo_id"].unique())
    print(f"val tomograms : {len(val_ids)}  (pos rows {len(pos_df)}, neg tomos {len(neg_ids)})")

    all_gt, all_hm, is_pos_flag = [], [], []

    # ---- POSITIVE slices ----
    # ---- POSITIVE slices ----
    print("\nPredicting on positive slices ...")
    for (tomo_id, z), group in pos_df.groupby(["tomo_id", "Motor axis 0"]):
        p = os.path.join(BASE_DIR, "train", tomo_id, f"slice_{int(z):04d}.jpg")
        if not os.path.exists(p):
            continue
        gt_list = [[int(r["Motor axis 1"]), int(r["Motor axis 2"])] for _, r in group.iterrows()]
        img_t = torch.from_numpy(load_norm(p)).unsqueeze(0).unsqueeze(0).to(DEVICE)

        # ---- U-Net boyut düzeltmesi (Padding) ----
        _, _, h, w = img_t.shape
        pad_h = (16 - h % 16) % 16
        pad_w = (16 - w % 16) % 16
        img_t = F.pad(img_t, (0, pad_w, 0, pad_h))
        # ------------------------------------------

        with torch.no_grad():
            hm = model(img_t).cpu().squeeze().numpy()
            hm = hm[:h, :w]  # Eklenen fazlalığı kırp ve orijinal boyuta dön
            all_hm.append(hm)
        all_gt.append(gt_list); is_pos_flag.append(True)

    # ---- NEGATIVE slices (yeni) ----
    # ---- NEGATIVE slices (yeni) ----
    print(f"Predicting on negative slices ({NEG_SLICES} per empty tomogram) ...")
    rng = np.random.default_rng(0)
    for tomo_id in neg_ids:
        d = os.path.join(BASE_DIR, "train", tomo_id)
        if not os.path.isdir(d):
            continue
        slices = sorted(f for f in os.listdir(d) if f.endswith(".jpg"))
        if not slices:
            continue
        lo, hi = int(0.3 * len(slices)), int(0.7 * len(slices))       # orta bolge
        cand = slices[lo:hi] or slices
        pick = rng.choice(cand, size=min(NEG_SLICES, len(cand)), replace=False)
        for fn_ in pick:
            img_t = torch.from_numpy(load_norm(os.path.join(d, fn_))).unsqueeze(0).unsqueeze(0).to(DEVICE)

            # ---- U-Net boyut düzeltmesi (Padding) ----
            _, _, h, w = img_t.shape
            pad_h = (16 - h % 16) % 16
            pad_w = (16 - w % 16) % 16
            img_t = F.pad(img_t, (0, pad_w, 0, pad_h))
            # ------------------------------------------

            with torch.no_grad():
                hm = model(img_t).cpu().squeeze().numpy()
                hm = hm[:h, :w]  # Eklenen fazlalığı kırp ve orijinal boyuta dön
                all_hm.append(hm)
            all_gt.append([]); is_pos_flag.append(False)

    # ---- threshold sweep ----
    print("\nThreshold sweep ...")
    thresholds = np.arange(0.05, 1.0, 0.05)
    rows = []
    for t in thresholds:
        dets = [detect_peaks(hm, threshold=t) for hm in all_hm]
        p, r, f1, tp, fp, fn = evaluate_detections(dets, all_gt)
        fp_neg = sum(len(d) for d, ip in zip(dets, is_pos_flag) if not ip)
        fp_pos = fp - fp_neg
        rows.append({"threshold": round(float(t), 2), "precision": round(p, 4),
                     "recall": round(r, 4), "f1": round(f1, 4),
                     "tp": tp, "fp": fp, "fn": fn,
                     "fp_on_neg_slices": fp_neg, "fp_on_pos_slices": fp_pos})
        print(f"  t={t:.2f}  P={p:.3f}  R={r:.3f}  F1={f1:.3f}  "
              f"TP={tp:4d} FP={fp:5d} (neg {fp_neg:4d}) FN={fn:4d}")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(os.path.join(RUN_DIR, "threshold_sweep.csv"), index=False)

    best = sweep.loc[sweep["f1"].idxmax()]
    print(f"\nBEST F1 = {best['f1']:.4f} @ threshold {best['threshold']:.2f}  "
          f"(P={best['precision']:.3f} R={best['recall']:.3f})")

    # ---- plots ----
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
    ax[0].plot(sweep["recall"], sweep["precision"], "b.-")
    ax[0].set_xlabel("Recall"); ax[0].set_ylabel("Precision")
    ax[0].set_title("Precision-Recall"); ax[0].grid(alpha=.3)
    ax[0].set_xlim(0, 1); ax[0].set_ylim(0, 1)

    ax[1].plot(sweep["threshold"], sweep["f1"], "r.-", label="F1")
    ax[1].plot(sweep["threshold"], sweep["precision"], "b.-", alpha=.5, label="P")
    ax[1].plot(sweep["threshold"], sweep["recall"], "g.-", alpha=.5, label="R")
    ax[1].axvline(best["threshold"], ls="--", c="gray", lw=1)
    ax[1].set_xlabel("Threshold"); ax[1].set_title("Metrics vs Threshold")
    ax[1].legend(); ax[1].grid(alpha=.3)

    ax[2].plot(sweep["threshold"], sweep["fp_on_neg_slices"], "m.-", label="FP on negative slices")
    ax[2].plot(sweep["threshold"], sweep["fp_on_pos_slices"], "c.-", label="FP on positive slices")
    ax[2].set_xlabel("Threshold"); ax[2].set_ylabel("False positives")
    ax[2].set_title("Where do FPs come from?"); ax[2].legend(); ax[2].grid(alpha=.3)

    fig.suptitle(f"{RUN_TAG}  |  hit_dist={HIT_DIST}px", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "precision_recall_metrics.png"), dpi=120)
    plt.close()

    # ---- save ----
    eval_metrics = {
        "best_f1": float(best["f1"]), "best_threshold": float(best["threshold"]),
        "precision_at_best": float(best["precision"]), "recall_at_best": float(best["recall"]),
        "tp": int(best["tp"]), "fp": int(best["fp"]), "fn": int(best["fn"]),
        "fp_on_neg_slices": int(best["fp_on_neg_slices"]),
        "fp_on_pos_slices": int(best["fp_on_pos_slices"]),
        "n_pos_slices": sum(is_pos_flag), "n_neg_slices": len(is_pos_flag) - sum(is_pos_flag),
        "hit_distance": HIT_DIST,
    }
    with open(os.path.join(RUN_DIR, "eval_metrics.json"), "w") as f:
        json.dump(eval_metrics, f, indent=2)

    csv_path = os.path.join(OUT_ROOT, "all_evals.csv")
    row = {"run_tag": RUN_TAG, **eval_metrics}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)

    print(f"\nSaved -> {RUN_DIR}")
    print(f"Appended -> {csv_path}")




