# Best run per model and starting condition (selected by F1 on val)

| condition | model | setting | F1 | AP | precision | recall | F1@thr0.5 | FP (no-motor) | median err (px) | train (min) | forward (ms) | params (M) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| From scratch | U-Net | sigma=14 | 0.766 | 0.706 | 0.901 | 0.667 | 0.76 | 3.0 | 4.26 | 50.5 | 22.754 | 8.114 |
| From scratch | YOLOv8m | box 64 px | 0.776 | 0.733 | 0.892 | 0.688 | 0.689 | 2.0 | 3.146 | 11.0 | 8.548 | 25.903 |
| From scratch | RT-DETR-L | lr0=0.0004, box 32 px, imgsz 640 | 0.5 | 0.421 | 0.65 | 0.406 | 0.479 | 0.0 | 3.83 | 19.94 | 17.596 | 32.97 |
| Pretrained | U-Net | - |  |  |  |  |  |  |  |  |  |  |
| Pretrained | YOLOv8m | box 48 px | 0.854 | 0.82 | 0.888 | 0.823 | 0.798 | 1.0 | 2.284 | 10.7 | 8.428 | 25.903 |
| Pretrained | RT-DETR-L | lr0=0.0002, box 48 px, imgsz 1024 | 0.849 | 0.835 | 0.916 | 0.792 | 0.849 | 1.0 | 2.828 | 24.31 | 18.579 | 32.97 |
