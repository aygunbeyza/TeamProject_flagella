"""Residual U-Net for heat-map regression (same architecture as before, + sigmoid output)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

PAD_MULTIPLE = 16          # 4 pooling steps -> input size must be divisible by 2**4


class ResBlock(nn.Module):
    def __init__(self, ci, co):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
            nn.Conv2d(co, co, 3, padding=1, bias=False), nn.BatchNorm2d(co))
        self.shortcut = (nn.Identity() if ci == co else
                         nn.Sequential(nn.Conv2d(ci, co, 1, bias=False), nn.BatchNorm2d(co)))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.body(x) + self.shortcut(x))


class UNet(nn.Module):
    """Output = heat map in [0, 1] (sigmoid on the last 1x1 convolution)."""

    def __init__(self, features=(32, 64, 128, 256)):
        super().__init__()
        features = list(features)
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(2)
        c = 1
        for f in features:
            self.downs.append(ResBlock(c, f))
            c = f
        self.bottleneck = ResBlock(features[-1], features[-1] * 2)
        for f in reversed(features):
            self.ups.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.ups.append(ResBlock(f * 2, f))
        self.final = nn.Conv2d(features[0], 1, 1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            x = self.ups[i + 1](torch.cat([skips[i // 2], x], dim=1))
        return torch.sigmoid(self.final(x))


def pad_to_multiple(x, multiple=PAD_MULTIPLE):
    """Zero-pad (right / bottom) a [..., H, W] tensor; returns (padded, (H, W)) with the original size."""
    h, w = x.shape[-2:]
    ph, pw = (-h) % multiple, (-w) % multiple
    return F.pad(x, (0, pw, 0, ph)), (h, w)


def weighted_mse(pred, target, hm_weight=10.0):
    """MSE with pixels around the motor weighted up to (1 + hm_weight) times."""
    weight = target * hm_weight + 1.0
    return torch.mean(weight * (pred - target) ** 2)
