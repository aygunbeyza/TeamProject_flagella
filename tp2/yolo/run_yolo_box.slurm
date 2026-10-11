#!/bin/bash
#SBATCH --job-name=yolo_box
#SBATCH --output=yolo/logs/yolo_box_%j.out
#SBATCH --error=yolo/logs/yolo_box_%j.err
#SBATCH --partition=alpha
#SBATCH --time=03:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023
# YOLO kutu boyutu taramasi (U-Net'teki sigma taramasinin karsiligi)
#   sbatch yolo/run_yolo_box.slurm              -> BOXES="32 64" (48 zaten var)
#   BOXES="32 64 80" SCRATCH=1 sbatch yolo/run_yolo_box.slurm
BOXES=${BOXES:-"32 64"}
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
FAILED=""
for B in $BOXES; do
  DS=processed/ultralytics_dataset_box${B}
  RUN=yolo/runs/${STEM}${SFX}_bg${BG}_ep${EPOCHS}_seed0_box${B}
  if [ -f "$RUN/eval_val/metrics.json" ]; then echo "[box $B] already done -> skipping"; continue; fi
  echo "=========== box $B ($(date)) run: $RUN ==========="
  if [ ! -f "$DS/val.txt" ]; then
    python common/export_yolo_dataset.py --box_px "$B" --out_dir "$DS" || { echo "EXPORT FAILED box $B"; FAILED="$FAILED $B"; continue; }
  fi
  python yolo/train_yolo.py --weights "$WEIGHTS" --bg_per_tomo "$BG" --epochs "$EPOCHS" \
      --dataset_dir "$DS" --tag "box${B}" $EXTRA \
    && python yolo/predict_yolo.py --run_dir "$RUN" --checkpoint last --dataset_dir "$DS" \
    && python evaluate.py --run_dir "$RUN" \
    || { echo "FAILED box $B"; FAILED="$FAILED $B"; }
done
python summarize_results.py --runs_dirs unet/runs yolo/runs --out_dir results_all || true
[ -n "$FAILED" ] && echo "!!! failed boxes:$FAILED"
echo "done -> yolo/runs/*_box*/eval_val/summary.txt"
