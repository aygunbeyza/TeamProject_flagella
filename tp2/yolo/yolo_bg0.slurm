#!/bin/bash
#SBATCH --job-name=yolo_bg0
#SBATCH --output=yolo/logs/yolo_bg0_%A_%a.out
#SBATCH --error=yolo/logs/yolo_bg0_%A_%a.err
#SBATCH --partition=alpha
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023
#SBATCH --array=0-4
# YOLOv8m without motor-free training slices (bg0). One array task = one run, all 5 run in PARALLEL.
#   0: pretrained box 32   1: pretrained box 64   2: scratch box 32   3: scratch box 48   4: scratch box 64
# (pretrained box 48 bg0 already exists)
module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env
source ~/yolo-venv/bin/activate
export FLAGELLA_DATA_DIR=/data/horse/ws/beay097h-teamproject_restore/beay097h-teamproject-1790463808/flagellar_motors_data
export PYTHONUNBUFFERED=1
export YOLO_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"

MODES=(pretrained pretrained scratch scratch scratch)
BOXES=(32 64 32 48 64)
MODE=${MODES[$SLURM_ARRAY_TASK_ID]}
BOX=${BOXES[$SLURM_ARRAY_TASK_ID]}
if [ "$MODE" = "scratch" ]; then SFX="_scratch"; EXTRA="--scratch"; else SFX=""; EXTRA=""; fi
if [ "$BOX" = "48" ]; then TAGSFX=""; TAGARG=""; DS=processed/ultralytics_dataset
else TAGSFX="_box$BOX"; TAGARG="--tag box$BOX"; DS=processed/ultralytics_dataset_box$BOX; fi
RUN=yolo/runs/yolov8m${SFX}_bg0_ep40_seed0${TAGSFX}
echo "=== $RUN  dataset: $DS"
if [ -f "$RUN/eval_val/metrics.json" ]; then echo "done already"; exit 0; fi
python yolo/train_yolo.py --weights yolo/weights/yolov8m.pt --bg_per_tomo 0 --epochs 40 --dataset_dir "$DS" $EXTRA $TAGARG || { echo "TRAIN FAILED"; exit 1; }
python yolo/predict_yolo.py --run_dir "$RUN" --checkpoint last || { echo "PREDICT FAILED"; exit 1; }
python evaluate.py --run_dir "$RUN" || { echo "EVAL FAILED"; exit 1; }
echo "done -> $RUN/eval_val/summary.txt"
