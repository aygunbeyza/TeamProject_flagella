import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

# --- CLUSTER PATHS ---
BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
MODEL_PATH = os.path.join(BASE_DIR, "unet_model.pth")
FIGURES_DIR = os.path.join("/data/horse/ws/beay097h-teamproject/TeamProject_flagella", "results")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. PEAK DETECTION FUNCTION ---
def detect_peaks(heatmap, threshold=0.3, min_distance=20):
    """Extract motor coordinates from heatmap."""
    local_max = maximum_filter(heatmap, size=min_distance)
    peaks_mask = (heatmap == local_max) & (heatmap >= threshold)
    
    ys, xs = np.where(peaks_mask)
    scores = heatmap[ys, xs]
    
    # Sort by score
    order = np.argsort(-scores)
    ys, xs, scores = ys[order], xs[order], scores[order]
    
    # NMS (Non-Maximum Suppression)
    keep = []
    for i in range(len(ys)):
        too_close = False
        for j in keep:
            dist = np.sqrt((ys[i]-ys[j])**2 + (xs[i]-xs[j])**2)
            if dist < min_distance:
                too_close = True
                break
        if not too_close:
            keep.append(i)
    return [(ys[i], xs[i], scores[i]) for i in keep]

# --- 2. LOADING MODEL ---
# Note: You need to import the UNet class from train_model or redefine it here
# For brevity in evaluation, ensure UNet class definition matches train_model.py
from train_model import UNet, MotorSliceDataset # Assuming you keep files in same dir

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# --- 3. RUN EVALUATION ON SAMPLE ---
# (Using the first training sample for verification)
# We assume train_samples list is accessible or reloaded
# For this script to work, ensure train_samples are defined or reloaded similarly to train_model.py

# --- 4. VISUALIZATION AND SAVING ---
print("Generating detection visualization...")
# ... (Your detection logic) ...

# Save the final plot
output_path = os.path.join(FIGURES_DIR, "final_detection_check.png")
plt.savefig(output_path)
print(f"Final detection visualization saved to: {output_path}")
