# YOLO

Every model (U-Net, YOLO, RT-DETR) must follow the SAME protocol:

1. Data: read `../processed/slices.csv` and `../processed/gt.csv` (made by `prepare_data.py`).
   Same train/val images, same voxel-spacing normalisation (`config.REFERENCE_SPACING`),
   same intensity normalisation (`common/imaging.py`), full images, no crop.
   `bg_rank < k` selects how many motor-free slices to train on (0 = motor slices only).
2. Epochs: same number as the U-Net (40), no early stopping, evaluate the LAST model.
3. Output goes to `yolo/runs/<run_tag>/`, e.g. `yolo/runs/yolov8m_bg3_ep40_seed0/`.
4. Required file: `predictions_val.csv` with columns `slice_id,y,x,score`
   (original image pixel coordinates, score in [0,1], keep detections down to `config.MIN_SCORE`).
5. Optional: `config.json`, `train_summary.json`, `inference_time.json` (same keys as the U-Net).

Then (from tp2/):
    python evaluate.py --run_dir yolo/runs/<run_tag>
    python summarize_results.py --runs_dirs unet/runs yolo/runs rtdetr/runs --out_dir results_all

Planned files: train_yolo.py, predict_yolo.py, run_yolo.slurm
