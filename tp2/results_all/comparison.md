# Results overview

Validation split `val` (motor + background slices, identical for every model), hit distance 24 px at reference spacing 15.6 (374.4 physical units).
F1 / F2 use the best threshold on the validation set (slightly optimistic).

| model | run_tag | F1 | F2 | P@F1 | R@F1 | AP | FP_background@F1 | epochs | train_min | fwd_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| rtdetr | rtdetr-l_bg3_ep40_seed0_lr1e-4_box48_img1024 | 0.328 | 0.426 | 0.244 | 0.500 | 0.214 | 1 | 40 | 24.360 | 18.597 |
| rtdetr | rtdetr-l_bg3_ep40_seed0_lr2e-4_box32_img1024 | 0.272 | 0.368 | 0.284 | 0.260 | 0.203 | 4 | 40 | 24.220 | 18.605 |
| rtdetr | rtdetr-l_bg3_ep40_seed0_lr2e-4_box48_img1024 | 0.849 | 0.819 | 0.916 | 0.792 | 0.835 | 1 | 40 | 24.310 | 18.579 |
| rtdetr | rtdetr-l_bg3_ep40_seed0_lr2e-4_box48_img640 | 0.789 | 0.765 | 0.873 | 0.719 | 0.776 | 1 | 40 | 19.010 | 17.356 |
| rtdetr | rtdetr-l_bg3_ep40_seed0_lr2e-4_box64_img1024 | 0.809 | 0.779 | 0.909 | 0.729 | 0.779 | 1 | 40 | 24.180 | 18.573 |
| rtdetr | rtdetr-l_bg3_ep40_seed0_lr4e-4_box48_img1024 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 0 | 40 | 24.220 | 18.501 |
| rtdetr | rtdetr-l_bg3_ep40_seed0_lr5e-5_box48_img1024 | 0.424 | 0.507 | 0.333 | 0.583 | 0.401 | 3 | 40 | 24.310 | 18.692 |
| rtdetr | rtdetr-l_scratch_bg3_ep40_seed0_lr1e-4_box48_img1024 | 0.135 | 0.195 | 0.109 | 0.177 | 0.084 | 11 | 40 | 24.980 | 18.592 |
| rtdetr | rtdetr-l_scratch_bg3_ep40_seed0_lr2e-4_box48_img1024 | 0.317 | 0.400 | 0.239 | 0.469 | 0.243 | 2 | 40 | 24.980 | 18.644 |
| rtdetr | rtdetr-l_scratch_bg3_ep40_seed0_lr4e-4_box32_img1024 | 0.333 | 0.441 | 0.259 | 0.469 | 0.309 | 1 | 40 | 25.830 | 18.754 |
| rtdetr | rtdetr-l_scratch_bg3_ep40_seed0_lr4e-4_box32_img640 | 0.500 | 0.485 | 0.650 | 0.406 | 0.421 | 0 | 40 | 19.940 | 17.596 |
| rtdetr | rtdetr-l_scratch_bg3_ep40_seed0_lr4e-4_box48_img1024 | 0.328 | 0.414 | 0.246 | 0.490 | 0.247 | 6 | 40 | 24.840 | 18.679 |
| rtdetr | rtdetr-l_scratch_bg3_ep40_seed0_lr4e-4_box64_img1024 | 0.261 | 0.356 | 0.252 | 0.271 | 0.184 | 0 | 40 | 26.290 | 18.837 |
| rtdetr | rtdetr-l_scratch_bg3_ep40_seed0_lr5e-5_box48_img1024 | 0.160 | 0.159 | 0.177 | 0.146 | 0.094 | 0 | 40 | 25.200 | 19.374 |
| unet | sigma12_bg0_ep40_seed0 | 0.678 | 0.636 | 0.773 | 0.604 | 0.600 | 12 | 40 | 20.600 | 23.102 |
| unet | sigma8_bg3_ep40_seed0 | 0.624 | 0.594 | 0.716 | 0.552 | 0.574 | 10 | 40 | 50.600 | 22.780 |
| unet | sigma10_bg3_ep40_seed0 | 0.745 | 0.681 | 0.923 | 0.625 | 0.653 | 2 | 40 | 50.500 | 22.740 |
| unet | sigma12_bg3_ep40_seed0 | 0.739 | 0.707 | 0.884 | 0.635 | 0.690 | 4 | 40 | 50.500 | 22.746 |
| unet | sigma14_bg3_ep40_seed0 | 0.766 | 0.708 | 0.901 | 0.667 | 0.706 | 3 | 40 | 50.500 | 22.754 |
| unet | sigma16_bg3_ep40_seed0 | 0.662 | 0.598 | 0.909 | 0.521 | 0.578 | 2 | 40 | 50.500 | 22.745 |
| unet | sigma18_bg3_ep40_seed0 | 0.671 | 0.588 | 0.911 | 0.531 | 0.561 | 3 | 40 | 50.500 | 22.749 |
| yolo | yolov8m_bg0_ep40_seed0 | 0.865 | 0.882 | 0.865 | 0.865 | 0.892 | 2 | 40 | 5.000 | 8.416 |
| yolo | yolov8m_bg3_ep40_seed0 | 0.854 | 0.839 | 0.888 | 0.823 | 0.820 | 1 | 40 | 10.700 | 8.428 |
| yolo | yolov8m_bg3_ep40_seed0_box32 | 0.841 | 0.849 | 0.828 | 0.854 | 0.836 | 4 | 40 | 10.600 | 8.329 |
| yolo | yolov8m_bg3_ep40_seed0_box64 | 0.783 | 0.743 | 0.929 | 0.677 | 0.731 | 3 | 40 | 10.600 | 8.415 |
| yolo | yolov8m_scratch_bg3_ep40_seed0 | 0.753 | 0.726 | 0.817 | 0.698 | 0.690 | 6 | 40 | 10.700 | 8.335 |
| yolo | yolov8m_scratch_bg3_ep40_seed0_box32 | 0.711 | 0.665 | 0.843 | 0.615 | 0.624 | 3 | 40 | 10.900 | 8.429 |
| yolo | yolov8m_scratch_bg3_ep40_seed0_box64 | 0.776 | 0.741 | 0.892 | 0.688 | 0.733 | 2 | 40 | 11.000 | 8.548 |

P@F1 / R@F1 = precision / recall at the best-F1 threshold; FP_background@F1 = false positives on background slices at that threshold; fwd_ms = network forward pass per image.
