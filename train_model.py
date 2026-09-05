import os, json, csv, datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset_builder import build_samples, MotorSliceDataset

# ==================================================================
# EXPERIMENT SETTINGS
# ==================================================================
SIGMA_VALUES = [8.0, 10.0, 12.0, 14.0, 16.0]
HIT_DIST     = 24
CHANGE_DESC  = "Sigma sweep (hit_distance sabit tutuldu)."
BASELINE_TAG = "run08_resnet_sigma12_hm10"

BG_PER_TOMO  = 3
HM_WEIGHT    = 10
PATCH        = None

LR         = 1e-4
BATCH_SIZE = 1            # farkli boyuttaki full-size goruntuler batch stack edilemedigi icin 1 yapildi
NUM_EPOCHS = 150
PATIENCE   = 10
FEATURES   = [32, 64, 128, 256]

# U-Net 4 kez /2 yaptigi icin 16 kullaniliyor
PAD_MULTIPLE = 16

# ==================================================================

BASE_DIR = "/data/horse/ws/beay097h-teamproject/flagellar_motors_data"
PROJ_DIR = "/data/horse/ws/beay097h-teamproject/TeamProject_flagella"
OUT_ROOT = os.path.join(PROJ_DIR, "output_2")

os.makedirs(OUT_ROOT, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================================================================
# BATCH PADDING
# ==================================================================
def pad_batch(batch, multiple=PAD_MULTIPLE):
    """
    Batch icindeki farkli H x W boyutundaki image/heatmap'leri
    ayni boyuta getirir.

    H ve W degerleri, batch'teki maksimum boyutun
    bir sonraki 16 katina yuvarlanir.

    Padding sadece sag ve alta eklenir.
    Motor koordinatlari degismez.
    """

    images = []
    heatmaps = []
    foreground_flags = []

    max_h = 0
    max_w = 0

    # ---------------- find maximum size ----------------
    for img, hm, is_foreground in batch:
        h = img.shape[-2]
        w = img.shape[-1]

        max_h = max(max_h, h)
        max_w = max(max_w, w)

    # ---------------- round to multiple of 16 ----------------
    target_h = int(np.ceil(max_h / multiple) * multiple)
    target_w = int(np.ceil(max_w / multiple) * multiple)

    # ---------------- pad each sample ----------------
    for img, hm, is_foreground in batch:

        h = img.shape[-2]
        w = img.shape[-1]

        pad_h = target_h - h
        pad_w = target_w - w

        # Image padding = 0
        img = F.pad(
            img,
            (0, pad_w, 0, pad_h),
            mode="constant",
            value=0
        )

        # Heatmap padding = 0
        hm = F.pad(
            hm,
            (0, pad_w, 0, pad_h),
            mode="constant",
            value=0
        )

        images.append(img)
        heatmaps.append(hm)
        foreground_flags.append(is_foreground)

    images = torch.stack(images, dim=0)
    heatmaps = torch.stack(heatmaps, dim=0)
    foreground_flags = torch.tensor(
        foreground_flags,
        dtype=torch.long
    )

    return images, heatmaps, foreground_flags


# ==================================================================
# MODEL
# ==================================================================
class ResBlock(nn.Module):

    def __init__(self, ci, co):
        super().__init__()

        self.b = nn.Sequential(
            nn.Conv2d(
                ci,
                co,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(co),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                co,
                co,
                3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(co)
        )

        self.shortcut = nn.Sequential()

        if ci != co:

            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    ci,
                    co,
                    1,
                    bias=False
                ),
                nn.BatchNorm2d(co)
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(
            self.b(x) + self.shortcut(x)
        )


class UNet(nn.Module):

    def __init__(self, feat=FEATURES):

        super().__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()

        self.pool = nn.MaxPool2d(2)

        c = 1

        # ---------------- encoder ----------------
        for f in feat:

            self.downs.append(
                ResBlock(c, f)
            )

            c = f

        # ---------------- bottleneck ----------------
        self.bottleneck = ResBlock(
            feat[-1],
            feat[-1] * 2
        )

        # ---------------- decoder ----------------
        for f in reversed(feat):

            self.ups.append(
                nn.ConvTranspose2d(
                    f * 2,
                    f,
                    2,
                    stride=2
                )
            )

            self.ups.append(
                ResBlock(
                    f * 2,
                    f
                )
            )

        self.final = nn.Conv2d(
            feat[0],
            1,
            1
        )

    def forward(self, x):

        skips = []

        # ---------------- encoder ----------------
        for d in self.downs:

            x = d(x)

            skips.append(x)

            x = self.pool(x)

        # ---------------- bottleneck ----------------
        x = self.bottleneck(x)

        # ---------------- decoder ----------------
        skips = skips[::-1]

        for i in range(
            0,
            len(self.ups),
            2
        ):

            x = self.ups[i](x)

            # Burada artik boyut mismatch beklenmez,
            # cunku input 16'nin katina pad edildi.
            x = self.ups[i + 1](
                torch.cat(
                    [
                        skips[i // 2],
                        x
                    ],
                    dim=1
                )
            )

        return self.final(x)


# ==================================================================
# LOSS
# ==================================================================
def weighted_mse(
    pred,
    hm,
    w=HM_WEIGHT
):

    weight = hm * w + 1.0

    return torch.mean(
        weight * (pred - hm) ** 2
    )


# ==================================================================
# EVALUATION HELPER
# ==================================================================
def prepare_single_image(img_t):
    """
    Tek goruntuyu model icin 16'nin katina pad eder.

    Original H/W degerlerini de dondurur.
    """

    _, h, w = img_t.shape

    target_h = int(
        np.ceil(h / PAD_MULTIPLE)
        * PAD_MULTIPLE
    )

    target_w = int(
        np.ceil(w / PAD_MULTIPLE)
        * PAD_MULTIPLE
    )

    pad_h = target_h - h
    pad_w = target_w - w

    padded = F.pad(
        img_t,
        (0, pad_w, 0, pad_h),
        mode="constant",
        value=0
    )

    return padded, h, w


# ==================================================================
# SINGLE EXPERIMENT
# ==================================================================
def run_experiment(
    sigma,
    train_ids,
    val_ids
):

    run_tag = (
        f"sigma{int(sigma):02d}_"
        f"hitdist{HIT_DIST}"
    )

    run_dir = os.path.join(
        OUT_ROOT,
        run_tag
    )

    os.makedirs(
        run_dir,
        exist_ok=True
    )

    print("=" * 70)
    print(f"RUN      : {run_tag}")
    print(f"CHANGE   : {CHANGE_DESC}")
    print(
        f"SIGMA    : {sigma}   "
        f"HIT_DIST : {HIT_DIST}  (fixed)"
    )
    print(f"OUTPUT   : {run_dir}")
    print(f"DEVICE   : {DEVICE}")

    if DEVICE == "cuda":

        print(
            f"GPU      : "
            f"{torch.cuda.get_device_name(0)}"
        )

    print(
        f"PADDING  : multiple of "
        f"{PAD_MULTIPLE}"
    )

    print("=" * 70)

    # ==================================================================
    # DATA
    # ==================================================================

    train_samples, train_fg, train_bg = build_samples(
        train_ids,
        bg_per_tomo=BG_PER_TOMO
    )

    val_samples, val_fg, val_bg = build_samples(
        val_ids,
        bg_per_tomo=BG_PER_TOMO
    )

    print(
        f"train  foreground={train_fg:5d}  "
        f"background={train_bg:5d}  "
        f"total={train_fg + train_bg:5d}"
    )

    print(
        f"val    foreground={val_fg:5d}  "
        f"background={val_bg:5d}  "
        f"total={val_fg + val_bg:5d}"
    )

    train_ds = MotorSliceDataset(
        train_samples,
        sigma=sigma,
        patch_size=PATCH
    )

    val_ds = MotorSliceDataset(
        val_samples,
        sigma=sigma,
        patch_size=PATCH
    )

    # IMPORTANT:
    # custom collate_fn ile farkli boyutlardaki image'leri
    # batch icinde 16'nin katina pad ediyoruz.

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        collate_fn=pad_batch
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        collate_fn=pad_batch
    )

    # ==================================================================
    # MODEL
    # ==================================================================

    model = UNet().to(DEVICE)

    opt = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )

    n_par = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        f"model params: {n_par:,}\n"
    )

    # ==================================================================
    # TRAIN
    # ==================================================================

    train_losses = []
    val_losses = []

    best_val = float("inf")
    best_ep = 0
    bad = 0

    t0 = datetime.datetime.now()

    for ep in range(
        1,
        NUM_EPOCHS + 1
    ):

        # ---------------- training ----------------
        model.train()

        tot = 0.0

        for img, hm, is_foreground in train_loader:

            img = img.to(DEVICE)
            hm = hm.to(DEVICE)

            opt.zero_grad()

            pred = model(img)

            loss = weighted_mse(
                pred,
                hm
            )

            loss.backward()

            opt.step()

            tot += loss.item()

        tr = tot / len(train_loader)

        # ---------------- validation ----------------
        model.eval()

        tot = 0.0

        with torch.no_grad():

            for img, hm, is_foreground in val_loader:

                img = img.to(DEVICE)
                hm = hm.to(DEVICE)

                pred = model(img)

                loss = weighted_mse(
                    pred,
                    hm
                )

                tot += loss.item()

        va = tot / len(val_loader)

        train_losses.append(tr)
        val_losses.append(va)

        print(
            f"Epoch {ep:03d}/{NUM_EPOCHS} | "
            f"Train {tr:.6f} | "
            f"Val {va:.6f}",
            end=""
        )

        if va < best_val:

            best_val = va
            best_ep = ep
            bad = 0

            torch.save(
                model.state_dict(),
                os.path.join(
                    run_dir,
                    "unet_model.pth"
                )
            )

            print(
                "  <- best saved"
            )

        else:

            bad += 1

            print(
                f"  (no improve "
                f"{bad}/{PATIENCE})"
            )

            if bad >= PATIENCE:

                print(
                    f"--- early stop "
                    f"at epoch {ep} ---"
                )

                break

    mins = (
        datetime.datetime.now()
        - t0
    ).total_seconds() / 60

    n_ep = len(train_losses)

    # ==================================================================
    # LOSS CURVE
    # ==================================================================

    plt.figure(
        figsize=(8, 5)
    )

    plt.plot(
        range(1, n_ep + 1),
        train_losses,
        label="Train"
    )

    plt.plot(
        range(1, n_ep + 1),
        val_losses,
        label="Val"
    )

    plt.axvline(
        best_ep,
        ls="--",
        c="gray",
        lw=1,
        label=f"best ep {best_ep}"
    )

    plt.xlabel("Epoch")

    plt.ylabel(
        f"Weighted MSE "
        f"(hm_w={HM_WEIGHT})"
    )

    plt.title(
        f"{run_tag}\n"
        f"{CHANGE_DESC}",
        fontsize=9
    )

    plt.legend()

    plt.grid(
        alpha=.3
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            run_dir,
            "loss_curve.png"
        ),
        dpi=120
    )

    plt.close()

    # ==================================================================
    # LOAD BEST MODEL
    # ==================================================================

    model.load_state_dict(
        torch.load(
            os.path.join(
                run_dir,
                "unet_model.pth"
            ),
            map_location=DEVICE
        )
    )

    model.eval()

    # ==================================================================
    # FOREGROUND EVALUATION
    # ==================================================================

    foreground_idx = [
        i
        for i, s in enumerate(val_samples)
        if s[3] == 1
    ]

    background_idx = [
        i
        for i, s in enumerate(val_samples)
        if s[3] == 0
    ]

    print(
        f"\nEvaluating "
        f"{len(foreground_idx)} "
        f"foreground val samples ..."
    )

    dists = []
    peaks = []
    hits = 0

    with torch.no_grad():

        for i in foreground_idx:

            img_t, hm_t, _ = val_ds[i]

            # ---------------- image padding ----------------
            img_pad, orig_h, orig_w = (
                prepare_single_image(img_t)
            )

            pred = model(
                img_pad.unsqueeze(0).to(DEVICE)
            )

            pred = pred.cpu().squeeze().numpy()

            # ---------------- crop back ----------------
            pred = pred[
                :orig_h,
                :orig_w
            ]

            tgt = hm_t.squeeze().numpy()

            ty, tx = np.unravel_index(
                tgt.argmax(),
                tgt.shape
            )

            py, px = np.unravel_index(
                pred.argmax(),
                pred.shape
            )

            d = float(
                np.hypot(
                    py - ty,
                    px - tx
                )
            )

            dists.append(d)

            peaks.append(
                float(pred.max())
            )

            if d <= HIT_DIST:
                hits += 1

    recall = (
        hits / len(foreground_idx)
        if foreground_idx
        else 0.0
    )

    # ==================================================================
    # BACKGROUND EVALUATION
    # ==================================================================

    background_peaks = []

    with torch.no_grad():

        for i in background_idx[:200]:

            img_t, _, _ = val_ds[i]

            img_pad, orig_h, orig_w = (
                prepare_single_image(img_t)
            )

            pred = model(
                img_pad.unsqueeze(0).to(DEVICE)
            )

            pred = pred.cpu().squeeze().numpy()

            pred = pred[
                :orig_h,
                :orig_w
            ]

            background_peaks.append(
                float(pred.max())
            )

    # ==================================================================
    # METRICS
    # ==================================================================

    metrics = {

        "sigma": sigma,

        "hit_distance": HIT_DIST,

        "n_foreground_eval":
            len(foreground_idx),

        "hits":
            hits,

        "recall_at_hitdist":
            round(recall, 4),

        "median_dist_px":
            (
                round(
                    float(np.median(dists)),
                    2
                )
                if dists
                else None
            ),

        "mean_peak_foreground":
            (
                round(
                    float(np.mean(peaks)),
                    4
                )
                if peaks
                else None
            ),

        "mean_peak_background":
            (
                round(
                    float(
                        np.mean(
                            background_peaks
                        )
                    ),
                    4
                )
                if background_peaks
                else None
            ),

        "best_val_loss":
            round(best_val, 6),

        "best_epoch":
            best_ep,

        "epochs_run":
            n_ep,

        "train_time_min":
            round(mins, 1)
    }

    print(
        json.dumps(
            metrics,
            indent=2
        )
    )

    # ==================================================================
    # PREDICTION FIGURE
    # ==================================================================

    if foreground_idx:

        pick = np.random.default_rng(
            0
        ).choice(
            foreground_idx,
            size=min(
                4,
                len(foreground_idx)
            ),
            replace=False
        )

        fig, ax = plt.subplots(
            len(pick),
            3,
            figsize=(
                12,
                4 * len(pick)
            )
        )

        if len(pick) == 1:
            ax = ax[None, :]

        with torch.no_grad():

            for r, i in enumerate(pick):

                img_t, hm_t, _ = val_ds[i]

                img_pad, orig_h, orig_w = (
                    prepare_single_image(img_t)
                )

                pred = model(
                    img_pad.unsqueeze(0)
                    .to(DEVICE)
                )

                pred = (
                    pred
                    .cpu()
                    .squeeze()
                    .numpy()
                )

                pred = pred[
                    :orig_h,
                    :orig_w
                ]

                img = (
                    img_t
                    .squeeze()
                    .numpy()
                )

                tgt = (
                    hm_t
                    .squeeze()
                    .numpy()
                )

                ty, tx = np.unravel_index(
                    tgt.argmax(),
                    tgt.shape
                )

                py, px = np.unravel_index(
                    pred.argmax(),
                    pred.shape
                )

                d = np.hypot(
                    py - ty,
                    px - tx
                )

                ok = d <= HIT_DIST

                ax[r, 0].imshow(
                    img,
                    cmap="gray"
                )

                ax[r, 0].plot(
                    tx,
                    ty,
                    "r+",
                    ms=14,
                    mew=2
                )

                ax[r, 0].set_title(
                    f"input "
                    f"(tomo "
                    f"{val_samples[i][0]})",
                    fontsize=8
                )

                ax[r, 1].imshow(
                    tgt,
                    cmap="hot"
                )

                ax[r, 1].set_title(
                    f"target "
                    f"sigma={sigma}",
                    fontsize=8
                )

                ax[r, 2].imshow(
                    pred,
                    cmap="hot"
                )

                ax[r, 2].plot(
                    tx,
                    ty,
                    "r+",
                    ms=14,
                    mew=2
                )

                ax[r, 2].plot(
                    px,
                    py,
                    "wx",
                    ms=12,
                    mew=2
                )

                ax[r, 2].set_title(
                    f"pred  "
                    f"d={d:.0f}px  "
                    f"max="
                    f"{pred.max():.3f}  "
                    f"{'HIT' if ok else 'MISS'}",
                    fontsize=8,
                    color=(
                        "green"
                        if ok
                        else "red"
                    )
                )

                for c in range(3):
                    ax[r, c].axis("off")

        fig.suptitle(
            f"{run_tag} | "
            f"{CHANGE_DESC}",
            fontsize=10
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                run_dir,
                "predictions.png"
            ),
            dpi=120
        )

        plt.close()

    # ==================================================================
    # SAVE CONFIG
    # ==================================================================

    config = {

        "run_tag": run_tag,

        "change_desc":
            CHANGE_DESC,

        "baseline_tag":
            BASELINE_TAG,

        "timestamp":
            datetime.datetime.now()
            .isoformat(
                timespec="seconds"
            ),

        "params": {

            "sigma":
                sigma,

            "hit_dist":
                HIT_DIST,

            "bg_per_tomo":
                BG_PER_TOMO,

            "hm_weight":
                HM_WEIGHT,

            "patch":
                PATCH,

            "lr":
                LR,

            "batch_size":
                BATCH_SIZE,

            "features":
                FEATURES,

            "num_epochs":
                NUM_EPOCHS,

            "patience":
                PATIENCE,

            "pad_multiple":
                PAD_MULTIPLE
        },

        "data": {

            "train_foreground":
                train_fg,

            "train_background":
                train_bg,

            "val_foreground":
                val_fg,

            "val_background":
                val_bg
        },

        "model_params":
            n_par
    }

    with open(
        os.path.join(
            run_dir,
            "config.json"
        ),
        "w"
    ) as f:

        json.dump(
            config,
            f,
            indent=2
        )

    with open(
        os.path.join(
            run_dir,
            "metrics.json"
        ),
        "w"
    ) as f:

        json.dump(
            metrics,
            f,
            indent=2
        )

    # ==================================================================
    # SUMMARY
    # ==================================================================

    with open(
        os.path.join(
            run_dir,
            "summary.txt"
        ),
        "w"
    ) as f:

        f.write(
            f"""{'=' * 66}
RUN      : {run_tag}
DATE     : {config['timestamp']}
CHANGE   : {CHANGE_DESC}
BASELINE : {BASELINE_TAG}
{'=' * 66}

PARAMETERS
  sigma        : {sigma}
  hit_dist     : {HIT_DIST} px
  bg_per_tomo  : {BG_PER_TOMO}
  hm_weight    : {HM_WEIGHT}
  patch        : {PATCH}
  features     : {FEATURES}
  lr           : {LR}
  batch_size   : {BATCH_SIZE}
  pad_multiple : {PAD_MULTIPLE}

DATA
  train : {train_fg} foreground / {train_bg} background
  val   : {val_fg} foreground / {val_bg} background

RESULTS
  recall @ {HIT_DIST}px : {recall:.3f} ({hits}/{len(foreground_idx)})
  median dist       : {metrics['median_dist_px']} px
  mean peak (fg)    : {metrics['mean_peak_foreground']}
  mean peak (bg)    : {metrics['mean_peak_background']} <- low is good
  best val loss     : {best_val:.6f} (epoch {best_ep})
  epochs run        : {n_ep}
  train time        : {mins:.1f} min

NOTE:
Input images are padded to a multiple of 16 before entering the U-Net.
Predictions are cropped back to the original image dimensions.
Hit distance is fixed across all sigma runs.
{'=' * 66}
"""
        )

    # ==================================================================
    # ALL RUNS CSV
    # ==================================================================

    csv_path = os.path.join(
        OUT_ROOT,
        "all_runs.csv"
    )

    row = {

        "run_tag":
            run_tag,

        "change":
            CHANGE_DESC,

        "sigma":
            sigma,

        "hit_dist":
            HIT_DIST,

        "bg_per_tomo":
            BG_PER_TOMO,

        "hm_weight":
            HM_WEIGHT,

        "patch":
            PATCH,

        "features":
            str(FEATURES),

        **metrics
    }

    write_header = not os.path.exists(
        csv_path
    )

    with open(
        csv_path,
        "a",
        newline=""
    ) as f:

        w = csv.DictWriter(
            f,
            fieldnames=list(
                row.keys()
            )
        )

        if write_header:
            w.writeheader()

        w.writerow(row)

    print(
        f"\nSaved -> {run_dir}"
    )

    print(
        f"Appended -> {csv_path}"
    )

    print(
        open(
            os.path.join(
                run_dir,
                "summary.txt"
            )
        ).read()
    )

    return (
        run_tag,
        run_dir,
        metrics
    )


# ==================================================================
# SWEEP ENTRY POINT
# ==================================================================
if __name__ == "__main__":

    with open(
        os.path.join(
            BASE_DIR,
            "train_ids.txt"
        )
    ) as f:

        train_ids = [
            line.strip()
            for line in f
            if line.strip()
        ]

    with open(
        os.path.join(
            BASE_DIR,
            "val_ids.txt"
        )
    ) as f:

        val_ids = [
            line.strip()
            for line in f
            if line.strip()
        ]

    all_results = []

    for sigma in SIGMA_VALUES:

        run_tag, run_dir, metrics = (
            run_experiment(
                sigma,
                train_ids,
                val_ids
            )
        )

        all_results.append(
            {
                "run_tag": run_tag,
                **metrics
            }
        )

    print(
        "\n" + "=" * 70
    )

    print(
        "SIGMA SWEEP COMPLETE"
    )

    print(
        "=" * 70
    )

    for r in all_results:

        print(
            f"  sigma={r['sigma']:5.1f}  "
            f"recall={r['recall_at_hitdist']:.3f}  "
            f"median_dist={r['median_dist_px']}px  "
            f"best_val_loss={r['best_val_loss']:.6f}  "
            f"-> {r['run_tag']}"
        )
