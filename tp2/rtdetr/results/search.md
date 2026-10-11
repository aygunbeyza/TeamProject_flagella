# RT-DETR parameter search (COCO-pretrained)

Fixed: 40 epochs, 3 motor-free slices per tomogram, batch 8, same validation set and metric as all models. Selection: best F1 on val (ties: AP).

| run | lr0 | box_px | imgsz | F1 | F1_thr | P@F1 | R@F1 | AP | F1@0.5 | FP_bg@F1 | best_epoch | train_min | selected |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rtdetr-l_bg3_ep40_seed0_lr1e-4_box48_img1024 | 1e-4 | 48 | 1024 | 0.328 | 0.4 | 0.244 | 0.5 | 0.214 | 0.103 | 1 | 34 | 24.36 |  |
| rtdetr-l_bg3_ep40_seed0_lr2e-4_box32_img1024 | 2e-4 | 32 | 1024 | 0.272 | 0.45 | 0.284 | 0.26 | 0.203 | 0.093 | 4 | 26 | 24.22 |  |
| rtdetr-l_bg3_ep40_seed0_lr2e-4_box48_img640 | 2e-4 | 48 | 640 | 0.789 | 0.7 | 0.873 | 0.719 | 0.776 | 0.764 | 1 | 38 | 19.01 |  |
| rtdetr-l_bg3_ep40_seed0_lr2e-4_box48_img1024 | 2e-4 | 48 | 1024 | 0.849 | 0.5 | 0.916 | 0.792 | 0.835 | 0.849 | 1 | 40 | 24.31 | <- BEST |
| rtdetr-l_bg3_ep40_seed0_lr2e-4_box64_img1024 | 2e-4 | 64 | 1024 | 0.809 | 0.7 | 0.909 | 0.729 | 0.779 | 0.787 | 1 | 40 | 24.18 |  |
| rtdetr-l_bg3_ep40_seed0_lr4e-4_box48_img1024 | 4e-4 | 48 | 1024 | 0.0 | 0.05 | 0.0 | 0.0 | 0.0 | 0.0 | 0 | 8 | 24.22 |  |
| rtdetr-l_bg3_ep40_seed0_lr5e-5_box48_img1024 | 5e-5 | 48 | 1024 | 0.424 | 0.4 | 0.333 | 0.583 | 0.401 | 0.06 | 3 | 36 | 24.31 |  |
