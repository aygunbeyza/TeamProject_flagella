# compare_train_val.py


import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from PIL import Image


# --- Import UNet and DEVICE from the training script ---
from train_model import UNet, DEVICE


# --- PATHS ---
BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
CSV_PATH = os.path.join(BASE_DIR, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
MODEL_PATH = os.path.join(BASE_DIR, "unet_model.pth")
FIGURES_DIR = os.path.join("/data/horse/ws/beay097h-teamproject/TeamProject_flagella", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)


# --- PEAK DETECTION (same as step 6) ---
def detect_peaks(heatmap, threshold=0.3, min_distance=20):
    """
    Find local maxima above a threshold with simple non-maximum suppression.
    Returns a list of (y, x, score).
    """
    local_max = maximum_filter(heatmap, size=min_distance)
    peaks_mask = (heatmap == local_max) & (heatmap >= threshold)


    ys, xs = np.where(peaks_mask)
    scores = heatmap[ys, xs]


    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]


    keep = []
    for i in range(len(ys)):
        too_close = False
        for j in keep:
            dist = np.sqrt((ys[i] - ys[j])**2 + (xs[i] - xs[j])**2)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            keep.append(i)
    return [(ys[i], xs[i], scores[i]) for i in keep]


# --- DETECTION EVALUATION (same as step 7) ---
def evaluate_detections(all_detections, all_ground_truths, hit_distance=30):
    """
    Compute TP, FP, FN, and derived precision/recall/F1 over a list of slices.
    all_detections: list of lists of (y, x, score)
    all_ground_truths: list of lists of [y, x]
    """
    tp, fp, fn = 0, 0, 0
    for dets, gt_list in zip(all_detections, all_ground_truths):
        gt_matched = [False] * len(gt_list)


        for (dy, dx, score) in dets:
            matched = False
            for i, gt in enumerate(gt_list):
                if gt_matched[i]:
                    continue
                dist = np.sqrt((dy - gt[0])**2 + (dx - gt[1])**2)
                if dist <= hit_distance:
                    tp += 1
                    gt_matched[i] = True
                    matched = True
                    break
            if not matched:
                fp += 1


        for matched in gt_matched:
            if not matched:
                fn += 1


    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


# --- RUN METRICS FOR ONE SPLIT (train or val) ---
def run_metrics_for_split(model, tomo_ids, split_name, thresholds=np.arange(0.1, 1.0, 0.05)):
    """
    For a given set of tomo_ids (train or validation), compute precision/recall/F1
    across a range of detection thresholds.
    Returns: thresholds, precisions, recalls, f1s
    """
    print(f"\n[{split_name}] computing metrics...")


    df = pd.read_csv(CSV_PATH)
    df_split = df[df["tomo_id"].isin(tomo_ids)]


    grouped = df_split.groupby(["tomo_id", "Motor axis 0"])


    all_ground_truths = []
    all_pred_heatmaps = []


    # 1) Generate ground truth coordinates and predicted heatmaps for all slices
    with torch.no_grad():
        for (tomo_id, z), group in grouped:
            img_path = os.path.join(TRAIN_DIR, tomo_id, f"slice_{int(z):04d}.jpg")
            if not os.path.exists(img_path):
                continue


            # Ground-truth coordinates (positive slices only)
            gt_list = []
            for _, r in group.iterrows():
                if r["Number of motors"] > 0:
                    gt_list.append([int(r["Motor axis 1"]), int(r["Motor axis 2"])])


            # Load and normalize the image (same normalization as training)
            img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
            lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
            img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)


            img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred_hm = model(img_t).cpu().squeeze().numpy()


            all_ground_truths.append(gt_list)
            all_pred_heatmaps.append(pred_hm)


    print(f"[{split_name}] number of slices used for metrics: {len(all_pred_heatmaps)}")


    # 2) Threshold sweep: Precision, Recall, F1
    precisions, recalls, f1s = [], [], []
    for t in thresholds:
        all_detections = []
        for hm in all_pred_heatmaps:
            dets = detect_peaks(hm, threshold=t, min_distance=20)
            all_detections.append(dets)


        p, r, f1 = evaluate_detections(all_detections, all_ground_truths, hit_distance=30)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)


    # Best F1 and corresponding threshold
    best_idx = int(np.argmax(f1s))
    best_thr = thresholds[best_idx]
    print(f"[{split_name}] best F1: {f1s[best_idx]:.4f} (threshold = {best_thr:.2f})")


    return thresholds, precisions, recalls, f1s


if __name__ == "__main__":
    # --- LOAD MODEL ---
    print("Loading model...")
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()


    # --- READ TRAIN AND VALIDATION IDs ---
    with open(os.path.join(BASE_DIR, "train_ids.txt")) as f:
        train_ids = set(l.strip() for l in f)
    with open(os.path.join(BASE_DIR, "val_ids.txt")) as f:
        val_ids = set(l.strip() for l in f)


    # --- METRICS FOR TRAIN ---
    th_train, prec_train, rec_train, f1_train = run_metrics_for_split(
        model, train_ids, split_name="Train"
    )


    # --- METRICS FOR VALIDATION ---
    th_val, prec_val, rec_val, f1_val = run_metrics_for_split(
        model, val_ids, split_name="Validation"
    )


    # --- COMPARISON PLOT: Train vs Validation ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))


    # Precision-Recall
    axes[0].plot(rec_train, prec_train, "b.-", label="Train")
    axes[0].plot(rec_val, prec_val, "r.-", label="Validation")
    axes[0].set_title("Precision-Recall (Train vs Validation)")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].legend()
    axes[0].grid(True)


    # F1 vs Threshold
    axes[1].plot(th_train, f1_train, "b.-", label="Train")
    axes[1].plot(th_val, f1_val, "r.-", label="Validation")
    axes[1].set_title("F1 Score vs Threshold (Train vs Validation)")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("F1 Score")
    axes[1].legend()
    axes[1].grid(True)


    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "train_vs_val_metrics.png")
    plt.savefig(out_path)
    print(f"\nTrain vs Validation metric plot saved to: {out_path}")


