#!/bin/bash
#SBATCH --job-name=yolo
#SBATCH --output=yolo/logs/yolo_%j.out
#SBATCH --error=yolo/logs/yolo_%j.err
#SBATCH --partition=alpha
#SBATCH --time=06:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023
# YOLO: egitim -> tahmin -> degerlendirme.   sbatch yolo/run_yolo.slurm
# Ayar: EPOCHS=40 BG=3 sbatch yolo/run_yolo.slurm     (BG=0 -> sadece motorlu slice)
#       SCRATCH=1 -> onceden egitilmis agirlik YOK (rastgele baslangic, U-Net ile adil kiyas)
EPOCHS=${EPOCHS:-40}
BG=${BG:-3}
WEIGHTS=${WEIGHTS:-yolo/weights/yolov8m.pt}
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
RUN=yolo/runs/${STEM}${SFX}_bg${BG}_ep${EPOCHS}_seed0
echo "host: $(hostname) date: $(date) epochs: $EPOCHS bg: $BG weights: $WEIGHTS scratch: $SCRATCH run: $RUN"

python yolo/train_yolo.py --weights "$WEIGHTS" --bg_per_tomo "$BG" --epochs "$EPOCHS" $EXTRA || { echo "TRAIN FAILED"; exit 1; }
python yolo/predict_yolo.py --run_dir "$RUN" --checkpoint last || { echo "PREDICT FAILED"; exit 1; }
python evaluate.py --run_dir "$RUN" || { echo "EVAL FAILED"; exit 1; }
python summarize_results.py --runs_dirs unet/runs yolo/runs --out_dir results_all || true
echo "done -> $RUN/eval_val/summary.txt"
