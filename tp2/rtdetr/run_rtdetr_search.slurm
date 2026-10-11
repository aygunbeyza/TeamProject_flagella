#!/bin/bash
#SBATCH --job-name=rtdetr_search
#SBATCH --output=rtdetr/logs/rtdetr_search_%j.out
#SBATCH --error=rtdetr/logs/rtdetr_search_%j.err
#SBATCH --partition=alpha
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023
# RT-DETR otomatik parametre aramasi (lr0 -> kutu boyutu -> imgsz), tablo + en iyi secimi.
#   sbatch rtdetr/run_rtdetr_search.slurm
#   SCRATCH=1 sbatch rtdetr/run_rtdetr_search.slurm      (sifirdan egitim, ayri tablo)
# Yeniden gondermek guvenli: biten kosular atlanir.
BATCH=${BATCH:-8}
SCRATCH=${SCRATCH:-0}
EXTRA=""
if [ "$SCRATCH" = "1" ]; then EXTRA="--scratch"; fi

module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env
source ~/yolo-venv/bin/activate
export FLAGELLA_DATA_DIR=/data/horse/ws/beay097h-teamproject_restore/beay097h-teamproject-1790463808/flagellar_motors_data
export PYTHONUNBUFFERED=1
export YOLO_OFFLINE=1
cd "$SLURM_SUBMIT_DIR"
echo "host: $(hostname) date: $(date) batch: $BATCH scratch: $SCRATCH"

python rtdetr/search_rtdetr.py --batch "$BATCH" $EXTRA
python summarize_results.py --runs_dirs unet/runs yolo/runs rtdetr/runs --out_dir results_all || true
echo "done -> rtdetr/results/search*.md  (en iyi: rtdetr/results/best*.json)"
