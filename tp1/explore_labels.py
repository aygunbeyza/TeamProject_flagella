import os
import pandas as pd
import numpy as np

# --- CLUSTER PATHS ---
BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
CSV_PATH = os.path.join(BASE_DIR, "train_labels.csv")

# Load labels
df = pd.read_csv(CSV_PATH)

print("Columns:", list(df.columns))
print("\nFirst rows:")
print(df.head())

# Basic tomogram statistics
tomo_stats = (
    df.groupby("tomo_id")
      .agg(
          n_rows=("Motor axis 0", "count"),
          n_motors=("Number of motors", "max")
      )
      .reset_index()
)

print("\nPer-tomogram stats (first 10):")
print(tomo_stats.head(10))

# Example: A tomogram with at least one motor (positive sample)
pos_rows = df[df["Number of motors"] > 0]
if not pos_rows.empty:
    ex = pos_rows.iloc[0]

    z = int(ex["Motor axis 0"])
    y = int(ex["Motor axis 1"])
    x = int(ex["Motor axis 2"])

    sz = int(ex["Array shape (axis 0)"])
    sy = int(ex["Array shape (axis 1)"])
    sx = int(ex["Array shape (axis 2)"])

    print("\nExample with a motor:")
    print(f"tomo_id: {ex['tomo_id']}")
    print(f"coords (z, y, x) = ({z}, {y}, {x})")
    print(f"array shape      = ({sz}, {sy}, {sx})")

    # Check if coordinates are within the volume boundaries
    inside = (0 <= z < sz) and (0 <= y < sy) and (0 <= x < sx)
    print(f"coords inside array shape? {inside}")

# Example: A tomogram with no motors (negative sample)
neg_rows = df[df["Number of motors"] == 0]
if not neg_rows.empty:
    ex_neg = neg_rows.iloc[0]
    
    print("\nExample with NO motor:")
    print(f"tomo_id: {ex_neg['tomo_id']}")
    print(
        "coords (z, y, x) =",
        int(ex_neg["Motor axis 0"]),
        int(ex_neg["Motor axis 1"]),
        int(ex_neg["Motor axis 2"]),
    )
    print(f"Number of motors: {int(ex_neg['Number of motors'])}")
