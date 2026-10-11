"""PyTorch dataset for the heat-map models (reads processed/slices.csv + gt.csv)."""
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset

from common.imaging import load_normalized, rescale_to_reference, gaussian_heatmap
from unet_model import PAD_MULTIPLE


class MotorSliceDataset(Dataset):
    """
    Returns (image[1,H,W], heatmap[1,H,W], has_motor).

    Pipeline per sample:  load + intensity normalisation -> rescale to the reference
    voxel spacing -> Gaussian heat map -> (training only) augmentation.

    Augmentation is ONLY applied if `augment=True` (training set); the validation
    set is therefore deterministic and identical in every run.
    """

    def __init__(self, slices_df, gt_df, sigma, reference_spacing, augment=False, aug_cfg=None, seed=0):
        self.rows = slices_df.to_dict("records")
        self.gt = {sid: g[["y", "x"]].to_numpy(np.float32) for sid, g in gt_df.groupby("slice_id")}
        self.sigma = sigma
        self.reference_spacing = reference_spacing
        self.augment = augment
        self.aug = dict(p_geometric=0.5, max_rotation_deg=15.0,
                        p_intensity=0.5, contrast_range=(0.8, 1.2), brightness_range=(-0.1, 0.1),
                        p_noise=0.5, noise_std_max=0.05)
        if aug_cfg:
            self.aug.update(aug_cfg)
        self.seed = seed

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        r = self.rows[idx]
        img = load_normalized(r["path"])
        img, sy, sx = rescale_to_reference(img, r["voxel_spacing"], self.reference_spacing)
        h, w = img.shape
        pts = [(y * sy, x * sx) for y, x in self.gt.get(r["slice_id"], [])]
        hm = gaussian_heatmap(h, w, pts, self.sigma)

        img_t = torch.from_numpy(np.ascontiguousarray(img))[None].float()
        hm_t = torch.from_numpy(hm)[None].float()

        if self.augment:
            # torch.initial_seed() differs per worker and per epoch -> different augmentation every time
            rng = np.random.default_rng((torch.initial_seed() + idx) % (2 ** 32))
            a = self.aug
            if rng.random() < a["p_geometric"]:                      # geometric: image AND heat map
                if rng.random() < 0.5:
                    img_t, hm_t = TF.hflip(img_t), TF.hflip(hm_t)
                if rng.random() < 0.5:
                    img_t, hm_t = TF.vflip(img_t), TF.vflip(hm_t)
                angle = float(rng.uniform(-a["max_rotation_deg"], a["max_rotation_deg"]))
                img_t, hm_t = TF.rotate(img_t, angle), TF.rotate(hm_t, angle)
            if rng.random() < a["p_intensity"]:                      # intensity: image only
                c = float(rng.uniform(*a["contrast_range"]))
                b = float(rng.uniform(*a["brightness_range"]))
                img_t = ((img_t - 0.5) * c + 0.5 + b).clamp(0.0, 1.0)
            if rng.random() < a["p_noise"]:                          # Gaussian noise: image only
                std = float(rng.uniform(0.0, a["noise_std_max"]))
                img_t = (img_t + std * torch.randn_like(img_t)).clamp(0.0, 1.0)

        return img_t, hm_t, int(r["has_motor"])


def pad_batch(batch, multiple=PAD_MULTIPLE):
    """Collate images of different size by zero-padding to the largest one (multiple of 16)."""
    max_h = max(b[0].shape[-2] for b in batch)
    max_w = max(b[0].shape[-1] for b in batch)
    th, tw = -(-max_h // multiple) * multiple, -(-max_w // multiple) * multiple
    imgs, hms, flags = [], [], []
    for img, hm, flag in batch:
        pad = (0, tw - img.shape[-1], 0, th - img.shape[-2])
        imgs.append(F.pad(img, pad))
        hms.append(F.pad(hm, pad))
        flags.append(flag)
    return torch.stack(imgs), torch.stack(hms), torch.tensor(flags, dtype=torch.long)
