import os
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter
from PIL import Image
from train_model import UNet, DEVICE # U-Net ve DEVICE'ı train_model'dan alıyoruz

# --- CLUSTER PATHS ---
BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
MODEL_PATH = os.path.join(BASE_DIR, "unet_model.pth")
FIGURES_DIR = os.path.join("/data/horse/ws/beay097h-teamproject/TeamProject_flagella", "results")

# --- 1. PEAK DETECTION FUNCTION (6. Adımdan Geldi) ---
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

# --- 2. EVALUATION FUNCTION ---
def evaluate_detections(all_detections, all_ground_truths, hit_distance=30):
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
                
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    return precision, recall, f1

if __name__ == "__main__":
    # --- 3. LOAD MODEL ---
    print("Model yükleniyor...")
    model = UNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # --- 4. PREPARE VALIDATION DATA (GÜNCELLENDİ) ---
    print("Doğrulama (Validation) seti hazırlanıyor...")
    with open(os.path.join(BASE_DIR, "val_ids.txt")) as f:
        val_ids = set(l.strip() for l in f)

    df = pd.read_csv(os.path.join(BASE_DIR, "train_labels.csv"))
    # Sadece val_ids içindeki tomogramları al (Pozitif veya negatif fark etmez)
    val_df = df[df["tomo_id"].isin(val_ids)]

    # Aynı tomogramın aynı kesitindeki motorları grupla
    grouped_val = val_df.groupby(["tomo_id", "Motor axis 0"])

    all_ground_truths = []
    all_pred_heatmaps = []

    # --- 5. GENERATE HEATMAPS FOR VALIDATION SET (GÜNCELLENDİ) ---
    print("Doğrulama seti üzerinden tahminler yapılıyor (Bu biraz sürebilir)...")
    with torch.no_grad():
        for (tomo_id, z), group in grouped_val:
            p = os.path.join(BASE_DIR, "train", tomo_id, f"slice_{int(z):04d}.jpg")
            if not os.path.exists(p):
                continue
                
            # Gerçek motor koordinatlarını topla (Boş resimler için liste boş kalacak)
            gt_list = []
            for _, r in group.iterrows():
                if r["Number of motors"] > 0:
                    gt_list.append([int(r["Motor axis 1"]), int(r["Motor axis 2"])])
            
            # Resmi yükle ve modele ver
            img = np.array(Image.open(p).convert("L"), dtype=np.float32)
            lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
            img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
            
            img_t = torch.from_numpy(img).unsqueeze(0).unsqueeze(0).to(DEVICE)
            pred_hm = model(img_t).cpu().squeeze().numpy()
            
            all_pred_heatmaps.append(pred_hm)
            all_ground_truths.append(gt_list)

    # --- 6. METRIC SWEEP ---
    print("Farklı eşik değerleri (thresholds) için F1 Skoru hesaplanıyor...")
    thresholds = np.arange(0.1, 1.0, 0.05)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        all_detections = []
        for hm in all_pred_heatmaps:
            dets = detect_peaks(hm, threshold=t, min_distance=20)
            all_detections.append(dets)
            
        p, r, f1 = evaluate_detections(all_detections, all_ground_truths)
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    # --- 7. PLOT AND SAVE ---
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(recalls, precisions, "b.-")
    axes[0].set_title("Precision-Recall Curve")
    axes[0].set_xlabel("Recall")
    axes[0].set_ylabel("Precision")
    axes[0].grid(True)

    axes[1].plot(thresholds, f1s, "r.-")
    axes[1].set_title("F1 Score vs Threshold")
    axes[1].set_xlabel("Threshold")
    axes[1].set_ylabel("F1 Score")
    axes[1].grid(True)

    plt.tight_layout()
    output_path = os.path.join(FIGURES_DIR, "precision_recall_metrics.png")
    plt.savefig(output_path)
    print(f"\nGrafikler başarıyla kaydedildi: {output_path}")

    best_idx = np.argmax(f1s)
    print(f"En iyi F1 Skoru: {f1s[best_idx]:.4f} (Eşik Değeri: {thresholds[best_idx]:.2f})")
