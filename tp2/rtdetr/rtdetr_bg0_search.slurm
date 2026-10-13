#!/bin/bash
#SBATCH --job-name=rtdetr_bg0s
#SBATCH --output=rtdetr/logs/rtdetr_bg0s_%A_%a.out
#SBATCH --error=rtdetr/logs/rtdetr_bg0s_%A_%a.err
#SBATCH --partition=alpha
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023
#SBATCH --array=0-4
# RT-DETR-L pretrained, without motor-free slices (bg0). One array task = one run (parallel).
#   0: lr 5e-5  1: lr 1e-4  2: lr 4e-4  (box 48)   3: box 32  4: box 64  (lr 2e-4)
module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env
source ~/yolo-venv/bin/activate
export FLAGELLA_DATA_DIR=/data/horse/ws/beay097h-teamproject_restore/beay097h-teamproject-1790463808/flagellar_motors_data
export PYTHONUNBUFFERED=1
export YOLO_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"

LRS=(5e-5 1e-4 4e-4 2e-4 2e-4)
BOXES=(48 48 48 32 64)
LR=${LRS[$SLURM_ARRAY_TASK_ID]}
BOX=${BOXES[$SLURM_ARRAY_TASK_ID]}
TAG=lr${LR}_box${BOX}_img1024
DS=processed/rtdetr_dataset_box$BOX
RUN=rtdetr/runs/rtdetr-l_bg0_ep40_seed0_$TAG
echo "=== $RUN  dataset: $DS"
if [ ! -d "$DS" ]; then echo "DATASET MISSING: $DS"; exit 1; fi
if [ -f "$RUN/eval_val/metrics.json" ]; then echo "done already"; exit 0; fi
python rtdetr/train_rtdetr.py --bg_per_tomo 0 --epochs 40 --lr0 $LR --imgsz 1024 --batch 8 --dataset_dir "$DS" --tag $TAG || { echo "TRAIN FAILED"; exit 1; }
python yolo/predict_yolo.py --model_type rtdetr --run_dir "$RUN" --checkpoint last || { echo "PREDICT FAILED"; exit 1; }
python evaluate.py --run_dir "$RUN" || { echo "EVAL FAILED"; exit 1; }
echo "done -> $RUN/eval_val/summary.txt"
