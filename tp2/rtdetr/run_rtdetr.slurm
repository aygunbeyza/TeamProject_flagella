#!/bin/bash
#SBATCH --job-name=rtdetr
#SBATCH --output=rtdetr/logs/rtdetr_%j.out
#SBATCH --error=rtdetr/logs/rtdetr_%j.err
#SBATCH --partition=alpha
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023
# RT-DETR: egitim -> tahmin -> degerlendirme.   sbatch rtdetr/run_rtdetr.slurm
# Test:  EPOCHS=1 sbatch rtdetr/run_rtdetr.slurm
# Ayar:  EPOCHS=40 BG=3 BATCH=8 SCRATCH=1 sbatch rtdetr/run_rtdetr.slurm
EPOCHS=${EPOCHS:-40}
BG=${BG:-3}
BATCH=${BATCH:-8}
WEIGHTS=${WEIGHTS:-rtdetr/weights/rtdetr-l.pt}
SCRATCH=${SCRATCH:-0}

module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env
source ~/yolo-venv/bin/activate
export FLAGELLA_DATA_DIR=/data/horse/ws/beay097h-teamproject_restore/beay097h-teamproject-1790463808/flagellar_motors_data
export PYTHONUNBUFFERED=1
export YOLO_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"

STEM=$(basename "$WEIGHTS" .pt)
EXTRA=""; SFX=""
if [ "$SCRATCH" = "1" ]; then EXTRA="--scratch"; SFX="_scratch"; fi
RUN=rtdetr/runs/${STEM}${SFX}_bg${BG}_ep${EPOCHS}_seed0
echo "host: $(hostname) date: $(date) epochs: $EPOCHS bg: $BG batch: $BATCH weights: $WEIGHTS scratch: $SCRATCH run: $RUN"

python rtdetr/train_rtdetr.py --weights "$WEIGHTS" --bg_per_tomo "$BG" --epochs "$EPOCHS" --batch "$BATCH" $EXTRA || { echo "TRAIN FAILED"; exit 1; }
python yolo/predict_yolo.py --model_type rtdetr --run_dir "$RUN" --checkpoint last || { echo "PREDICT FAILED"; exit 1; }
python evaluate.py --run_dir "$RUN" || { echo "EVAL FAILED"; exit 1; }
python summarize_results.py --runs_dirs unet/runs yolo/runs rtdetr/runs --out_dir results_all || true
echo "done -> $RUN/eval_val/summary.txt"
