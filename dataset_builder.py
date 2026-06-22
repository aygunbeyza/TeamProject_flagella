import os
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
from torch.utils.data import Dataset

# --- CLUSTER PATHS ---
BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
CSV_PATH = os.path.join(BASE_DIR, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
FIGURES_DIR = os.path.join("/data/horse/ws/beay097h-teamproject/TeamProject_flagella", "results")

os.makedirs(FIGURES_DIR, exist_ok=True)
SIGMA = 3.0  # Gaussian sigma in pixels

# ---------- Utilities ----------
def load_and_normalize(path):
    """Load grayscale jpg and normalize using 0.5/99.5 percentiles."""
    img = np.array(Image.open(path).convert("L"), dtype=np.float32)
    low, high = np.percentile(img, 0.5), np.percentile(img, 99.5)
    img = np.clip((img - low) / (high - low + 1e-8), 0, 1)
    return img  # (H, W) in [0,1]

def make_gaussian_heatmap(h, w, y, x, sigma=3.0):
    """Return (H, W) heatmap with a 2D Gaussian peak at (y, x)."""
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dist2 = (yy - y) ** 2 + (xx - x) ** 2
    heatmap = np.exp(-dist2 / (2 * sigma ** 2)).astype(np.float32)
    return heatmap  # (H, W)

# ---------- Build list of samples from CSV ----------
df = pd.read_csv(CSV_PATH)
samples = []  # each item: (image_path, y, x)

for _, row in df.iterrows():
    if row["Number of motors"] <= 0:
        continue  # skip rows with no motor

    tomo_id = row["tomo_id"]
    z = int(row["Motor axis 0"])
    y = int(row["Motor axis 1"])
    x = int(row["Motor axis 2"])

    tomo_dir = os.path.join(TRAIN_DIR, tomo_id)
    img_path = os.path.join(tomo_dir, f"slice_{z:04d}.jpg")

    if os.path.exists(img_path):
        samples.append((img_path, y, x))

print(f"Total samples with existing slice: {len(samples)}")
if len(samples) == 0:
    raise RuntimeError("No (image, motor) pairs found on disk.")

# ---------- PyTorch Dataset ----------
class MotorSliceDataset(Dataset):
    """2D dataset: (normalized slice, Gaussian heatmap)."""

    def __init__(self, samples, sigma=3.0):
        self.samples = samples
        self.sigma = sigma

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, y, x = self.samples[idx]

        img = load_and_normalize(img_path)         # (H, W)
        h, w = img.shape

        # clamp coordinates to image bounds
        y_clamp = int(np.clip(y, 0, h - 1))
        x_clamp = int(np.clip(x, 0, w - 1))

        heatmap = make_gaussian_heatmap(h, w, y_clamp, x_clamp, self.sigma)

        # add channel dim and convert to tensors
        img_t = torch.from_numpy(img).unsqueeze(0)       # (1, H, W)
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0)  # (1, H, W)

        return img_t, heatmap_t

# ---------- Quick visual check ----------
if __name__ == "__main__":
    dataset = MotorSliceDataset(samples, sigma=SIGMA)
    print(f"Dataset size: {len(dataset)}")

    # Visualize the first sample and save it
    img_t, hm_t = dataset[0]
    img = img_t.squeeze(0).numpy()
    heatmap = hm_t.squeeze(0).numpy()

    plt.figure(figsize=(10, 4))
    
    plt.subplot(1, 2, 1)
    plt.title("Input slice")
    plt.imshow(img, cmap="gray")
    plt.axis("off")
    
    plt.subplot(1, 2, 2)
    plt.title("Target heatmap")
    plt.imshow(img, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.4)
    plt.axis("off")
    
    plt.tight_layout()
    
    # Save the combined check plot
    output_path = os.path.join(FIGURES_DIR, "dataset_check_sample0.png")
    plt.savefig(output_path)
    print(f"Visual check saved to: {output_path}")

    # --- NEW: Heatmap only check ---
    plt.figure(figsize=(4, 4))
    plt.title("Heatmap only")
    plt.imshow(heatmap, cmap="jet")
    plt.colorbar()
    plt.axis("off")
    
    heatmap_output_path = os.path.join(FIGURES_DIR, "heatmap_only_sample0.png")
    plt.savefig(heatmap_output_path)
    print(f"Heatmap check saved to: {heatmap_output_path}")
    print(f"Max heatmap value: {heatmap.max()}")
