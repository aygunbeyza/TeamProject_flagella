# RT-DETR parameter search (from scratch)

Fixed: 40 epochs, 3 motor-free slices per tomogram, batch 8, same validation set and metric as all models. Selection: best F1 on val (ties: AP).

| run | lr0 | box_px | imgsz | F1 | F1_thr | P@F1 | R@F1 | AP | F1@0.5 | FP_bg@F1 | best_epoch | train_min | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtdetr-l_scratch_bg3_ep40_seed0_lr1e-4_box48_img1024 | 1e-4 | 48 | 1024 | 0.135 | 0.3 | 0.109 | 0.177 | 0.084 | 0.0 | 11 | 36 | 24.98 |  |
| rtdetr-l_scratch_bg3_ep40_seed0_lr2e-4_box48_img1024 | 2e-4 | 48 | 1024 | 0.317 | 0.4 | 0.239 | 0.469 | 0.243 | 0.268 | 2 | 39 | 24.98 |  |
| rtdetr-l_scratch_bg3_ep40_seed0_lr4e-4_box32_img640 | 4e-4 | 32 | 640 | 0.5 | 0.55 | 0.65 | 0.406 | 0.421 | 0.479 | 0 | 38 | 19.94 | <- BEST |
| rtdetr-l_scratch_bg3_ep40_seed0_lr4e-4_box32_img1024 | 4e-4 | 32 | 1024 | 0.333 | 0.45 | 0.259 | 0.469 | 0.309 | 0.273 | 1 | 39 | 25.83 |  |
| rtdetr-l_scratch_bg3_ep40_seed0_lr4e-4_box48_img1024 | 4e-4 | 48 | 1024 | 0.328 | 0.45 | 0.246 | 0.49 | 0.247 | 0.274 | 6 | 40 | 24.84 |  |
| rtdetr-l_scratch_bg3_ep40_seed0_lr4e-4_box64_img1024 | 4e-4 | 64 | 1024 | 0.261 | 0.45 | 0.252 | 0.271 | 0.184 | 0.074 | 0 | 32 | 26.29 |  |
| rtdetr-l_scratch_bg3_ep40_seed0_lr5e-5_box48_img1024 | 5e-5 | 48 | 1024 | 0.16 | 0.25 | 0.177 | 0.146 | 0.094 | 0.0 | 0 | 38 | 25.2 |  |
