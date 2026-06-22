import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from train_model import UNet, MotorSliceDataset, DEVICE # Reuse your model classes

# --- CLUSTER PATHS ---
BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
MODEL_PATH = os.path.join(BASE_DIR, "unet_model.pth")
FIGURES_DIR = os.path.join("/data/horse/ws/beay097h-teamproject/TeamProject_flagella", "results")

# --- EVALUATION FUNCTION ---
def evaluate_detections(all_detections, all_ground_truths, hit_distance=30):
    tp, fp, fn = 0, 0, 0
    for dets, gt in zip(all_detections, all_ground_truths):
        gt_matched = False
        for (dy, dx, score) in dets:
            dist = np.sqrt((dy - gt[0])**2 + (dx - gt[1])**2)
            if dist <= hit_distance and not gt_matched:
                tp += 1
                gt_matched = True
            else:
                fp += 1
        if not gt_matched:
            fn += 1
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

# --- LOAD MODEL ---
model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- METRIC SWEEP ---
# (In a real run, you would iterate over the whole val_loader here)
print("Calculating PR Curve and optimal threshold...")

thresholds = np.arange(0.1, 1.0, 0.05)
precisions, recalls, f1s = [], [], []

# [Insert your logic to generate detections for val_loader here]
# ...

# --- PLOT AND SAVE ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(recalls, precisions, "b.-")
axes[0].set_title("Precision-Recall Curve")
axes[0].grid(True)

axes[1].plot(thresholds, f1s, "r.-")
axes[1].set_title("F1 Score vs Threshold")
axes[1].grid(True)

plt.tight_layout()
output_path = os.path.join(FIGURES_DIR, "precision_recall_metrics.png")
plt.savefig(output_path)
print(f"Metrics plot saved to: {output_path}")

best_idx = np.argmax(f1s)
print(f"\nBest F1: {f1s[best_idx]:.4f} at Threshold: {thresholds[best_idx]:.2f}")
