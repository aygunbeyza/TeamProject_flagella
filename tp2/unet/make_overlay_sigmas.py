import sys, numpy as np, torch, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import predict_unet as P

cfg = P.config
sid = sys.argv[1]
dev = "cuda" if torch.cuda.is_available() else "cpu"
sl = pd.read_csv(cfg.SLICES_CSV)
r = sl[sl.slice_id == sid].iloc[0]
img = P.load_normalized(r.path)
img_r, sy, sx = P.rescale_to_reference(img, r.voxel_spacing, cfg.REFERENCE_SPACING)

for s in (8, 14):
    run = f"unet/runs/sigma{s}_bg3_ep40_seed0"
    ckpt = torch.load(run + "/model_last.pth", map_location=dev)
    model = P.UNet(ckpt["features"]).to(dev)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    x, (h, w) = P.pad_to_multiple(torch.from_numpy(np.ascontiguousarray(img_r))[None, None])
    with torch.no_grad():
        hm = model(x.to(dev))[0, 0, :h, :w].cpu().numpy()
    print("sigma", s, "heat map max", float(hm.max()))
    print("   peaks (y, x, score):", [(int(a), int(b), round(float(c), 2)) for a, b, c in P.detect_peaks(hm, cfg.MIN_SCORE, 20)])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(img_r, cmap="gray")
    ax.imshow(hm, cmap="jet", vmin=0, vmax=1, alpha=np.clip(hm * 1.6, 0, 0.85))
    ax.axis("off")
    out = f"results_final/unet_overlay_{sid}_sigma{s}.png"
    plt.savefig(out, dpi=200, bbox_inches="tight", pad_inches=0)
    plt.close()
    print("saved", out)
