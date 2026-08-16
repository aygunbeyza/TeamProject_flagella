import os, json, csv, datetime
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_builder import build_samples, MotorSliceDataset

# ==================================================================
#  EXPERIMENT SETTINGS  --  change ONLY this block per run
# ==================================================================
RUN_TAG      = "run05_sigma12_hm20"
CHANGE_DESC  = "HM_WEIGHT 100'den 20'ye dusuruldu."
BASELINE_TAG = "run03_sigma_12"

NEG_PER_TOMO = 3         # Negatif oranımız (boş tomogram başına 3 kesit)
HM_WEIGHT    = 20
SIGMA        = 12.0
PATCH        = 512
HIT_DIST     = int(2 * SIGMA) #it will be 2*sigma

LR         = 1e-4
BATCH_SIZE = 8
NUM_EPOCHS = 150
PATIENCE   = 10
FEATURES   = [32, 64, 128, 256]
# ==================================================================

BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
PROJ_DIR = "/data/horse/ws/beay097h-teamproject/TeamProject_flagella"
OUT_ROOT = os.path.join(PROJ_DIR, "output")
RUN_DIR  = os.path.join(OUT_ROOT, RUN_TAG)
os.makedirs(RUN_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("=" * 70)
print(f"RUN      : {RUN_TAG}")
print(f"CHANGE   : {CHANGE_DESC}")
print(f"BASELINE : {BASELINE_TAG}")
print(f"OUTPUT   : {RUN_DIR}")
print(f"DEVICE   : {DEVICE}")
if DEVICE == "cuda":
    print(f"GPU      : {torch.cuda.get_device_name(0)}")
print("=" * 70)


# ---------------- model ----------------
class Block(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.b = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    def forward(self, x): return self.b(x)


class UNet(nn.Module):
    def __init__(self, feat=FEATURES):
        super().__init__()
        self.downs, self.ups = nn.ModuleList(), nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        c = 1
        for f in feat:
            self.downs.append(Block(c, f)); c = f
        self.bottleneck = Block(feat[-1], feat[-1] * 2)
        for f in reversed(feat):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(Block(f * 2, f))
        self.final = nn.Conv2d(feat[0], 1, 1)

    def forward(self, x):
        skips = []
        for d in self.downs:
            x = d(x); skips.append(x); x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = self.ups[i + 1](torch.cat([skips[i // 2], x], dim=1))
        return self.final(x)


def weighted_mse(pred, hm, w=HM_WEIGHT):
    weight = hm * w + 1.0
    return torch.mean(weight * (pred - hm) ** 2)



if __name__ == "__main__":

# ---------------- data ----------------
    with open(os.path.join(BASE_DIR, "train_ids.txt")) as f:
        train_ids = [line.strip() for line in f if line.strip()]
        
    with open(os.path.join(BASE_DIR, "val_ids.txt")) as f:
        val_ids = [line.strip() for line in f if line.strip()]

    train_samples, tp, tn = build_samples(train_ids, neg_per_tomo=NEG_PER_TOMO)

    val_samples,   vp, vn = build_samples(val_ids,   neg_per_tomo=NEG_PER_TOMO)
    print(f"train  pos={tp:5d}  neg={tn:5d}  total={tp+tn:5d}")
    print(f"val    pos={vp:5d}  neg={vn:5d}  total={vp+vn:5d}")

    train_ds = MotorSliceDataset(train_samples, sigma=SIGMA, patch_size=PATCH)
    val_ds   = MotorSliceDataset(val_samples,   sigma=SIGMA, patch_size=PATCH)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    model = UNet().to(DEVICE)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_par:,}\n")


    # ---------------- train ----------------
    train_losses, val_losses = [], []
    best_val, best_ep, bad = float("inf"), 0, 0
    t0 = datetime.datetime.now()

    for ep in range(1, NUM_EPOCHS + 1):
        model.train(); tot = 0.0
        for img, hm, is_pos in train_loader:
            img, hm = img.to(DEVICE), hm.to(DEVICE)
            opt.zero_grad()
            loss = weighted_mse(model(img), hm)
            loss.backward(); opt.step()
            tot += loss.item()
        tr = tot / len(train_loader)

        model.eval(); tot = 0.0
        with torch.no_grad():
            for img, hm, is_pos in val_loader:
                img, hm = img.to(DEVICE), hm.to(DEVICE)
                tot += weighted_mse(model(img), hm).item()
        va = tot / len(val_loader)

        train_losses.append(tr); val_losses.append(va)
        print(f"Epoch {ep:03d}/{NUM_EPOCHS} | Train {tr:.6f} | Val {va:.6f}", end="")

        if va < best_val:
            best_val, best_ep, bad = va, ep, 0
            torch.save(model.state_dict(), os.path.join(RUN_DIR, "unet_model.pth"))
            print("  <- best saved")
        else:
            bad += 1
            print(f"  (no improve {bad}/{PATIENCE})")
            if bad >= PATIENCE:
                print(f"--- early stop at epoch {ep} ---")
                break

    mins = (datetime.datetime.now() - t0).total_seconds() / 60
    n_ep = len(train_losses)

    # ---------------- loss curve ----------------
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, n_ep + 1), train_losses, label="Train")
    plt.plot(range(1, n_ep + 1), val_losses,   label="Val")
    plt.axvline(best_ep, ls="--", c="gray", lw=1, label=f"best ep {best_ep}")
    plt.xlabel("Epoch"); plt.ylabel(f"Weighted MSE (hm_w={HM_WEIGHT})")
    plt.title(f"{RUN_TAG}\n{CHANGE_DESC}", fontsize=9)
    plt.legend(); plt.grid(alpha=.3); plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR, "loss_curve.png"), dpi=120); plt.close()


    # ---------------- evaluate ----------------
    model.load_state_dict(torch.load(os.path.join(RUN_DIR, "unet_model.pth")))
    model.eval()

    pos_idx = [i for i, s in enumerate(val_samples) if s[3] == 1]
    neg_idx = [i for i, s in enumerate(val_samples) if s[3] == 0]

    print(f"\nEvaluating {len(pos_idx)} positive val samples ...")
    dists, peaks, hits = [], [], 0
    with torch.no_grad():
        for i in pos_idx:
            img_t, hm_t, _ = val_ds[i]
            pred = model(img_t.unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()
            tgt  = hm_t.squeeze().numpy()
            ty, tx = np.unravel_index(tgt.argmax(),  tgt.shape)
            py, px = np.unravel_index(pred.argmax(), pred.shape)
            d = float(np.hypot(py - ty, px - tx))
            dists.append(d); peaks.append(float(pred.max()))
            if d <= HIT_DIST: hits += 1

    recall = hits / len(pos_idx) if pos_idx else 0.0

    neg_peaks = []
    with torch.no_grad():
        for i in neg_idx[:200]:
            img_t, _, _ = val_ds[i]
            pred = model(img_t.unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()
            neg_peaks.append(float(pred.max()))

    metrics = {
        "n_pos_eval":        len(pos_idx),
        "hits":              hits,
        "recall_at_hitdist": round(recall, 4),
        "median_dist_px":    round(float(np.median(dists)), 2) if dists else None,
        "mean_peak_pos":     round(float(np.mean(peaks)), 4) if peaks else None,
        "mean_peak_neg":     round(float(np.mean(neg_peaks)), 4) if neg_peaks else None,
        "best_val_loss":     round(best_val, 6),
        "best_epoch":        best_ep,
        "epochs_run":        n_ep,
        "train_time_min":    round(mins, 1),
    }
    print(json.dumps(metrics, indent=2))


    # ---------------- prediction figure ----------------
    pick = np.random.default_rng(0).choice(pos_idx, size=min(4, len(pos_idx)), replace=False)
    fig, ax = plt.subplots(len(pick), 3, figsize=(12, 4 * len(pick)))
    if len(pick) == 1: ax = ax[None, :]
    with torch.no_grad():
        for r, i in enumerate(pick):
            img_t, hm_t, _ = val_ds[i]
            pred = model(img_t.unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()
            img  = img_t.squeeze().numpy(); tgt = hm_t.squeeze().numpy()
            ty, tx = np.unravel_index(tgt.argmax(),  tgt.shape)
            py, px = np.unravel_index(pred.argmax(), pred.shape)
            d  = np.hypot(py - ty, px - tx)
            ok = d <= HIT_DIST

            ax[r, 0].imshow(img, cmap="gray"); ax[r, 0].plot(tx, ty, "r+", ms=14, mew=2)
            ax[r, 0].set_title(f"input  (tomo {val_samples[i][0]})", fontsize=8)
            ax[r, 1].imshow(tgt, cmap="hot"); ax[r, 1].set_title(f"target  sigma={SIGMA}", fontsize=8)
            ax[r, 2].imshow(pred, cmap="hot")
            ax[r, 2].plot(tx, ty, "r+", ms=14, mew=2)
            ax[r, 2].plot(px, py, "wx", ms=12, mew=2)
            ax[r, 2].set_title(f"pred  d={d:.0f}px  max={pred.max():.3f}  "
                               f"{'HIT' if ok else 'MISS'}",
                               fontsize=8, color="green" if ok else "red")
            for c in range(3): ax[r, c].axis("off")
    fig.suptitle(f"{RUN_TAG} | {CHANGE_DESC}", fontsize=10)
    plt.tight_layout(); plt.savefig(os.path.join(RUN_DIR, "predictions.png"), dpi=120); plt.close()


    # ---------------- save records ----------------
    config = {
        "run_tag": RUN_TAG, "change_desc": CHANGE_DESC, "baseline_tag": BASELINE_TAG,
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "params": {"neg_per_tomo": NEG_PER_TOMO, "hm_weight": HM_WEIGHT, "sigma": SIGMA,
                   "patch": PATCH, "hit_dist": HIT_DIST, "lr": LR,
                   "batch_size": BATCH_SIZE, "features": FEATURES,
                   "num_epochs": NUM_EPOCHS, "patience": PATIENCE},
        "data": {"train_pos": tp, "train_neg": tn, "val_pos": vp, "val_neg": vn},
        "model_params": n_par,
    }
    with open(os.path.join(RUN_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    with open(os.path.join(RUN_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    with open(os.path.join(RUN_DIR, "summary.txt"), "w") as f:
        f.write(f"""{'='*66}
    RUN      : {RUN_TAG}
    DATE     : {config['timestamp']}
    CHANGE   : {CHANGE_DESC}
    BASELINE : {BASELINE_TAG}
    {'='*66}

    PARAMETERS
      neg_per_tomo : {NEG_PER_TOMO}   <- the only changed variable
      hm_weight    : {HM_WEIGHT}
      sigma        : {SIGMA}
      patch        : {PATCH}
      hit_dist     : {HIT_DIST} px
      features     : {FEATURES}
      lr           : {LR}
      batch_size   : {BATCH_SIZE}

    DATA
      train : {tp} pos / {tn} neg
      val   : {vp} pos / {vn} neg

    RESULTS
      recall @ {HIT_DIST}px : {recall:.3f}   ({hits}/{len(pos_idx)})
      median dist       : {metrics['median_dist_px']} px
      mean peak (pos)   : {metrics['mean_peak_pos']}
      mean peak (neg)   : {metrics['mean_peak_neg']}   <- low is good
      best val loss     : {best_val:.6f}  (epoch {best_ep})
      epochs run        : {n_ep}
      train time        : {mins:.1f} min

    NOTE: hm_weight is unchanged, so val loss IS comparable to the baseline run.
    {'='*66}
    """)

    csv_path = os.path.join(OUT_ROOT, "all_runs.csv")
    row = {"run_tag": RUN_TAG, "change": CHANGE_DESC,
           "neg_per_tomo": NEG_PER_TOMO, "hm_weight": HM_WEIGHT, "sigma": SIGMA,
           "patch": PATCH, "hit_dist": HIT_DIST, "features": str(FEATURES),
           **metrics}
    write_header = not os.path.exists(csv_path)
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header: w.writeheader()
        w.writerow(row)

    print(f"\nSaved -> {RUN_DIR}")
    print(f"Appended -> {csv_path}")
    print(open(os.path.join(RUN_DIR, "summary.txt")).read())
