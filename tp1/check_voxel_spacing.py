"""
train_labels.csv icinde 'voxel spacing' ile ilgili sutunu bulur ve istatistiklerini gosterir.
"""
import pandas as pd
import os

BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
CSV_PATH = os.path.join(BASE_DIR, "train_labels.csv")

df = pd.read_csv(CSV_PATH)
print("Tum sutunlar:")
print(list(df.columns))

candidates = [c for c in df.columns if "spac" in c.lower() or "voxel" in c.lower()]
print(f"\n'spacing'/'voxel' iceren sutunlar: {candidates}")

for col in candidates:
    print(f"\n--- {col} ---")
    print(df[col].describe())
    print(f"En kucuk: {df[col].min()}   En buyuk: {df[col].max()}")
    print(f"En buyuk / en kucuk orani: {df[col].max() / df[col].min():.2f}x")
