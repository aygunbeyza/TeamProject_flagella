import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd

# --- CLUSTER PATHS ---
BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
CSV_PATH = os.path.join(BASE_DIR, "train_labels.csv")
TOMO_ID = "tomo_00e047"
TOMO_DIR = os.path.join(BASE_DIR, "train", TOMO_ID)

# Create directory for saving figures
FIGURES_DIR = os.path.join("/data/horse/ws/beay097h-teamproject/TeamProject_flagella", "results")
os.makedirs(FIGURES_DIR, exist_ok=True)

# Load labels
df = pd.read_csv(CSV_PATH)

# Filter rows with motors for the specific tomogram
df_tomo = df[(df["tomo_id"] == TOMO_ID) & (df["Number of motors"] > 0)]
if df_tomo.empty:
    raise ValueError(f"No motor rows found for {TOMO_ID}")

# Find an existing slice on disk
img_path = None
row_used = None

for _, row in df_tomo.iterrows():
    z = int(row["Motor axis 0"])
    candidate = os.path.join(TOMO_DIR, f"slice_{z:04d}.jpg")
    if os.path.exists(candidate):
        img_path = candidate
        row_used = row
        break

if img_path is None:
    raise FileNotFoundError(f"No matching slice_XXXX.jpg found in {TOMO_DIR}")

print(f"Using image: {img_path}")

# Extract motor coordinates
z = int(row_used["Motor axis 0"])
y = int(row_used["Motor axis 1"])  # row (y)
x = int(row_used["Motor axis 2"])  # column (x)

# Load and normalize image
img = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
low, high = np.percentile(img, 0.5), np.percentile(img, 99.5)
img_norm = np.clip((img - low) / (high - low + 1e-8), 0, 1)

# Plot and save the figure
plt.figure(figsize=(6, 6))
plt.imshow(img_norm, cmap="gray")
plt.scatter(x, y, s=100, c="red", marker="x", label="motor")
plt.title(f"{TOMO_ID}, slice {z} | Motor: x={x}, y={y}")
plt.legend()
plt.axis("off")
plt.tight_layout()

# Save with descriptive filename
filename = f"motor_target_{TOMO_ID}_slice_{z:04d}_x{x}_y{y}.png"
output_image_path = os.path.join(FIGURES_DIR, filename)

plt.savefig(output_image_path)
print(f"Figure saved successfully: {output_image_path}")
