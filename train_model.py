import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# --- CLUSTER PATHS ---
BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
CSV_PATH = os.path.join(BASE_DIR, "train_labels.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train")
FIGURES_DIR = os.path.join("/data/horse/ws/beay097h-teamproject/TeamProject_flagella", "results")

# Check GPU availability
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("ther e is no gpu.")


# --- 1. DATASET DEFINITION ---
class MotorSliceDataset(Dataset):
    def __init__(self, samples, sigma=6.0, patch_size=512):
        self.samples = samples
        self.sigma = sigma
        self.ps = patch_size

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, y, x = self.samples[idx]
        
        img = np.array(Image.open(path).convert("L"), dtype=np.float32)
        lo, hi = np.percentile(img, 0.5), np.percentile(img, 99.5)
        img = np.clip((img - lo) / (hi - lo + 1e-8), 0, 1)
        h, w = img.shape
        y, x = int(np.clip(y, 0, h-1)), int(np.clip(x, 0, w-1))

        if self.ps and self.ps < min(h, w):
            # 1. calculate limits
            t_min = max(0, y - self.ps + 1)
            t_max = min(h - self.ps, y)
            l_min = max(0, x - self.ps + 1)
            l_max = min(w - self.ps, x)

            # 2. choose eandom
            t = np.random.randint(t_min, max(t_min, t_max) + 1)
            l = np.random.randint(l_min, max(l_min, l_max) + 1)

            # 3. cut and calculate coordinates again
            img = img[t:t+self.ps, l:l+self.ps]
            y, x = y - t, x - l

        ph, pw = img.shape
        yy, xx = np.meshgrid(np.arange(ph), np.arange(pw), indexing="ij")
        hm = np.exp(-((yy-y)**2 + (xx-x)**2) / (2*self.sigma**2)).astype(np.float32)

        return torch.from_numpy(img).unsqueeze(0), torch.from_numpy(hm).unsqueeze(0)

# --- 2. U-NET ARCHITECTURE ---
class Block(nn.Module):
    def __init__(self, inc, outc):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(inc, outc, 3, padding=1), nn.BatchNorm2d(outc), nn.ReLU(True),
            nn.Conv2d(outc, outc, 3, padding=1), nn.BatchNorm2d(outc), nn.ReLU(True)
        )
    def forward(self, x): return self.net(x)

class UNet(nn.Module):
    def __init__(self, feat=[32, 64, 128, 256]):
        super().__init__()
        self.encs = nn.ModuleList()
        self.decs = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        
        inc = 1
        for f in feat:
            self.encs.append(Block(inc, f))
            inc = f
        self.bot = Block(feat[-1], feat[-1]*2)
        
        for f in reversed(feat):
            self.decs.append(nn.ConvTranspose2d(f*2, f, 2, 2))
            self.decs.append(Block(f*2, f))
        self.head = nn.Conv2d(feat[0], 1, 1)

    def forward(self, x):
        skips = []
        for enc in self.encs:
            x = enc(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bot(x)
        skips = skips[::-1]
        for i in range(0, len(self.decs), 2):
            x = self.decs[i](x)
            s = skips[i//2]
            if x.shape != s.shape:
                x = nn.functional.interpolate(x, s.shape[2:])
            x = self.decs[i+1](torch.cat([s, x], 1))
        return torch.sigmoid(self.head(x))

# --- 3. TRAINING LOOP ---
if __name__ == "__main__":
    print("Loading Train/Val IDs...")
    with open(os.path.join(BASE_DIR, "train_ids.txt")) as f:
        train_ids = set(l.strip() for l in f)
    with open(os.path.join(BASE_DIR, "val_ids.txt")) as f:
        val_ids = set(l.strip() for l in f)

    def get_samples(tomo_ids):
        df = pd.read_csv(CSV_PATH)
        df = df[(df["Number of motors"] > 0) & (df["tomo_id"].isin(tomo_ids))]
        samples = []
        for _, r in df.iterrows():
            p = os.path.join(TRAIN_DIR, r["tomo_id"], f"slice_{int(r['Motor axis 0']):04d}.jpg")
            if os.path.exists(p):
                samples.append((p, int(r["Motor axis 1"]), int(r["Motor axis 2"])))
        return samples

    train_samples = get_samples(train_ids)
    val_samples = get_samples(val_ids)
    print(f"Train samples: {len(train_samples)}, Val samples: {len(val_samples)}")

    # Loaders optimized for Cluster (batch_size=8, num_workers=4)
    train_loader = DataLoader(MotorSliceDataset(train_samples), batch_size=8, shuffle=True, num_workers=4)
    val_loader = DataLoader(MotorSliceDataset(val_samples), batch_size=8, shuffle=False, num_workers=4)

    print("Initializing model and optimizer...")
    model = UNet().to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    
    # --- EARLY STOPPING VE EPOCH AYARLARI ---
    NUM_EPOCHS = 150  # Maksimum sınır (Early stopping daha önce kesecek)
    patience = 10     # 10 epoch boyunca iyileşme olmazsa durdur
    best_val_loss = float("inf")
    epochs_no_improve = 0
    train_losses, val_losses = [], [] 

    print("Starting Training...")
    for ep in range(1, NUM_EPOCHS + 1):
        # Train phase
        model.train()
        tloss = 0
        for img, hm in train_loader:
            img, hm = img.to(DEVICE), hm.to(DEVICE)
            pred = model(img)
            
            # YENİ: Ağırlıklı MSE Loss
            weight = hm * 100 + 1.0  
            loss = torch.mean(weight * (pred - hm)**2)
            
            opt.zero_grad()
            loss.backward()
            opt.step()
            
            tloss += loss.item()
        tloss /= max(len(train_loader), 1)

        # Validation phase
        model.eval()
        vloss = 0
        with torch.no_grad():
            for img, hm in val_loader:
                img, hm = img.to(DEVICE), hm.to(DEVICE)
                pred_val = model(img) # Tahmini hesapla
                
                # YENİ: Doğrulama için Ağırlıklı MSE Loss
                weight_val = hm * 100 + 1.0
                vloss += torch.mean(weight_val * (pred_val - hm)**2).item()
        vloss /= max(len(val_loader), 1)

        train_losses.append(tloss)
        val_losses.append(vloss)
        print(f"Epoch {ep:02d}/{NUM_EPOCHS} | Train Loss: {tloss:.6f} | Val Loss: {vloss:.6f}")

        # --- EARLY STOPPING & CHECKPOINT KONTROLÜ ---
        model_save_path = os.path.join(BASE_DIR, "unet_model.pth")
        
        if vloss < best_val_loss:
            best_val_loss = vloss
            epochs_no_improve = 0
            # Sadece model geliştiğinde ağırlıkları kaydet
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> Validation Loss düştü! En iyi model kaydedildi.")
        else:
            epochs_no_improve += 1
            print(f"  -> Gelişme yok. (Sabır: {epochs_no_improve}/{patience})")
            
            if epochs_no_improve >= patience:
                print(f"\n--- Early Stopping tetiklendi! Eğitimi durduruluyor. ---")
                break

    # --- SAVE THE LOSS PLOT ---
    plt.figure(figsize=(8, 5))
    # Gerçekleşen epoch sayısına göre x eksenini ayarla (Early stopping için önemli)
    actual_epochs = len(train_losses) 
    plt.plot(range(1, actual_epochs + 1), train_losses, label="Train Loss")
    plt.plot(range(1, actual_epochs + 1), val_losses, label="Val Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    
    loss_plot_path = os.path.join(FIGURES_DIR, "training_loss_curve.png")
    plt.savefig(loss_plot_path)
    print(f"Loss curve saved to: {loss_plot_path}")

# --- 4. VISUALIZE PREDICTIONS ---
    print("Generating prediction samples...")
    # Modelin en iyi kaydedilmiş halini yükle (Opsiyonel ama daha güvenilir tahminler için iyi bir adım)
    model.load_state_dict(torch.load(model_save_path))
    model.eval()
    
    # plot 3 sample with doğrulama ile
    dataset_val = MotorSliceDataset(val_samples if len(val_samples) > 0 else train_samples)
    n = min(3, len(dataset_val))
    
    fig, axes = plt.subplots(n, 3, figsize=(12, 4*n))
    if n == 1: 
        axes = axes[np.newaxis, :]
        
    for i in range(n):
        img_t, hm_t = dataset_val[i]
        
        # we do prediction with model here
        with torch.no_grad():
            pred = model(img_t.unsqueeze(0).to(DEVICE)).cpu().squeeze().numpy()
            
        img = img_t.squeeze().numpy()
        tgt = hm_t.squeeze().numpy()

        # 1. column:original input
        axes[i,0].imshow(img, cmap="gray")
        axes[i,0].set_title("Input")
        axes[i,0].axis("off")
        
        # 2. column:Target
        axes[i,1].imshow(img, cmap="gray")
        axes[i,1].imshow(tgt, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        axes[i,1].set_title("Target")
        axes[i,1].axis("off")
        
        # 3.column:  Prediction
        axes[i,2].imshow(img, cmap="gray")
        axes[i,2].imshow(pred, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        axes[i,2].set_title("Prediction")
        axes[i,2].axis("off")

    plt.tight_layout()
    pred_plot_path = os.path.join(FIGURES_DIR, "prediction_samples.png")
    plt.savefig(pred_plot_path)
    print(f"Prediction samples saved successfully to: {pred_plot_path}")
