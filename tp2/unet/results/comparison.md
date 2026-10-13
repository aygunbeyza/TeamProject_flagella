# Results overview

Validation split `val` (motor + background slices, identical for every model), hit distance 24 px at reference spacing 15.6 (374.4 physical units).
F1 / F2 use the best threshold on the validation set (slightly optimistic).

| model | run_tag | F1 | F2 | P@F1 | R@F1 | AP | FP_background@F1 | epochs | train_min | fwd_ms |
|---|---|---|---|---|---|---|---|---|---|---|
| unet | sigma10_bg0_ep40_seed0 | 0.646 | 0.587 | 0.779 | 0.552 | 0.584 | 9 | 40 | 20.400 | 22.907 |
| unet | sigma12_bg0_ep40_seed0 | 0.678 | 0.636 | 0.773 | 0.604 | 0.600 | 12 | 40 | 20.600 | 23.102 |
| unet | sigma14_bg0_ep40_seed0 | 0.667 | 0.615 | 0.818 | 0.562 | 0.598 | 9 | 40 | 20.600 | 22.928 |
| unet | sigma8_bg3_ep40_seed0 | 0.624 | 0.594 | 0.716 | 0.552 | 0.574 | 10 | 40 | 50.600 | 22.780 |
| unet | sigma10_bg3_ep40_seed0 | 0.745 | 0.681 | 0.923 | 0.625 | 0.653 | 2 | 40 | 50.500 | 22.740 |
| unet | sigma12_bg3_ep40_seed0 | 0.739 | 0.707 | 0.884 | 0.635 | 0.690 | 4 | 40 | 50.500 | 22.746 |
| unet | sigma14_bg3_ep40_seed0 | 0.766 | 0.708 | 0.901 | 0.667 | 0.706 | 3 | 40 | 50.500 | 22.754 |
| unet | sigma16_bg3_ep40_seed0 | 0.662 | 0.598 | 0.909 | 0.521 | 0.578 | 2 | 40 | 50.500 | 22.745 |
| unet | sigma18_bg3_ep40_seed0 | 0.671 | 0.588 | 0.911 | 0.531 | 0.561 | 3 | 40 | 50.500 | 22.749 |

P@F1 / R@F1 = precision / recall at the best-F1 threshold; FP_background@F1 = false positives on background slices at that threshold; fwd_ms = network forward pass per image.
