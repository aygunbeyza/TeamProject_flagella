import os
import numpy as np
import pandas as pd

# --- CLUSTER PATHS ---
BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
CSV_PATH = os.path.join(BASE_DIR, "train_labels.csv")

VAL_FRACTION = 0.2
RANDOM_SEED = 42

# Load labels
df = pd.read_csv(CSV_PATH)

# Get unique tomogram IDs
tomo_ids = np.array(sorted(df["tomo_id"].unique()))
print(f"Total tomograms: {len(tomo_ids)}")

# Random train/val split by tomo_id
rng = np.random.default_rng(RANDOM_SEED)
perm = rng.permutation(len(tomo_ids))
val_size = max(1, int(len(tomo_ids) * VAL_FRACTION))

val_ids = tomo_ids[perm[:val_size]]
train_ids = tomo_ids[perm[val_size:]]

print(f"Train tomograms: {len(train_ids)}")
print(f"Val tomograms  : {len(val_ids)}")
print(f"\nExample train IDs: {train_ids[:5]}")
print(f"Example val IDs  : {val_ids[:5]}")

# Save ID lists to text files
train_file = os.path.join(BASE_DIR, "train_ids.txt")
val_file = os.path.join(BASE_DIR, "val_ids.txt")

with open(train_file, "w") as f:
    for t in train_ids:
        f.write(f"{t}\n")

with open(val_file, "w") as f:
    for t in val_ids:
        f.write(f"{t}\n")

print(f"\nSaved: {train_file}")
print(f"Saved: {val_file}")

# Sanity checks
assert set(train_ids).isdisjoint(val_ids)
assert set(train_ids) | set(val_ids) == set(tomo_ids)
print("\nSanity checks passed! Split is valid and mutually exclusive.")
