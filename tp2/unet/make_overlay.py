import sys, numpy as np, torch, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import predict_unet as P

cfg = P.config
run = "unet/runs/sigma14_bg3_ep40_seed0"
dev = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(run + "/model_last.pth", map_location=dev)
model = P.UNet(ckpt["features"]).to(dev)
model.load_state_dict(ckpt["state_dict"])
model.eval()

sl = pd.read_csv(cfg.SLICES_CSV)
sl = sl[sl.split == "val"]
gt = pd.read_csv(cfg.GT_CSV)

if len(sys.argv) < 2:
    g = gt.merge(sl[["slice_id", "height", "width"]], on="slice_id")
    ok = g[(g.y >= 50) & (g.y <= g.height - 50) & (g.x >= 50) & (g.x <= g.width - 50)]
    ids = np.random.default_rng(0).permutation(ok.slice_id.unique())[:4]
    print("candidate slice ids:", list(ids))
    print("usage: python unet/make_overlay.py <slice_id>")
    sys.exit()

sid = sys.argv[1]
r = sl[sl.slice_id == sid].iloc[0]
img = P.load_normalized(r.path)
img_r, sy, sx = P.rescale_to_reference(img, r.voxel_spacing, cfg.REFERENCE_SPACING)
x, (h, w) = P.pad_to_multiple(torch.from_numpy(np.ascontiguousarray(img_r))[None, None])
with torch.no_grad():
    out = model(x.to(dev))
hm = out[0, 0, :h, :w].cpu().numpy()
pts = [(yy * sy, xx * sx) for yy, xx in gt.loc[gt.slice_id == sid, ["y", "x"]].to_numpy()]
print("slice", sid, "heat map max", float(hm.max()), "motor (y,x)", pts)

cy, cx = pts[0]
y0 = int(max(0, min(cy - 190, h - 380)))
x0 = int(max(0, min(cx - 300, w - 600)))
im = img_r[y0:y0 + 380, x0:x0 + 600]
hc = hm[y0:y0 + 380, x0:x0 + 600]
fig, ax = plt.subplots(figsize=(6.3, 4))
ax.imshow(im, cmap="gray")
ax.imshow(hc, cmap="jet", vmin=0, vmax=1, alpha=np.clip(hc * 1.6, 0, 0.85))
for (yy, xx) in pts:
    ax.plot(xx - x0, yy - y0, "c+", ms=18, mew=2)
ax.axis("off")
plt.savefig("results_final/unet_heatmap_overlay.png", dpi=200, bbox_inches="tight", pad_inches=0)
print("saved results_final/unet_heatmap_overlay.png")
