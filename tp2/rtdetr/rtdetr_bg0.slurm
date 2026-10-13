#!/bin/bash
#SBATCH --job-name=rtdetr_bg0
#SBATCH --output=rtdetr/logs/rtdetr_bg0_%j.out
#SBATCH --error=rtdetr/logs/rtdetr_bg0_%j.err
#SBATCH --partition=alpha
#SBATCH --time=01:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023
# RT-DETR-L pretrained, best bg3 setting (lr0 2e-4, box 48, imgsz 1024), WITHOUT motor-free training slices (bg0)
module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env
source ~/yolo-venv/bin/activate
export FLAGELLA_DATA_DIR=/data/horse/ws/beay097h-teamproject_restore/beay097h-teamproject-1790463808/flagellar_motors_data
export PYTHONUNBUFFERED=1
export YOLO_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
RUN=rtdetr/runs/rtdetr-l_bg0_ep40_seed0_lr2e-4_box48_img1024
python rtdetr/train_rtdetr.py --bg_per_tomo 0 --epochs 40 --lr0 2e-4 --imgsz 1024 --batch 8 --dataset_dir processed/rtdetr_dataset_box48 --tag lr2e-4_box48_img1024 || { echo "TRAIN FAILED"; exit 1; }
python yolo/predict_yolo.py --model_type rtdetr --run_dir "$RUN" --checkpoint last || { echo "PREDICT FAILED"; exit 1; }
python evaluate.py --run_dir "$RUN" || { echo "EVAL FAILED"; exit 1; }
echo "done -> $RUN/eval_val/summary.txt"
