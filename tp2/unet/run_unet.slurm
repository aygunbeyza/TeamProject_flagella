#!/bin/bash
#SBATCH --job-name=unet_sigma_sweep
#SBATCH --output=unet/logs/unet_%j.out
#SBATCH --error=unet/logs/unet_%j.err
#SBATCH --partition=alpha
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023
# =============================================================================
# U-NET: TEK SLURM DOSYASI -> sbatch unet/run_unet.slurm
#
# Kullanım: tp2 klasörüne gir ve çalıştır:  cd .../tp2 && sbatch unet/run_unet.slurm
# 1) veriyi hazırlar (processed/ yoksa)             : prepare_data.py
# 2) HER sigma için: eğitim -> tahmin -> değerlendirme (motorlu + motorsuz slice ile)
# 3) sigma karşılaştırma tablosu + grafikler        : unet/results/
#
# Ayar (isteğe bağlı):  SIGMAS="10 12" EPOCHS=40 sbatch unet/run_unet.slurm
# Yeniden başlatılabilir: değerlendirmesi bitmiş sigma'lar atlanır (süre dolarsa tekrar gönder).
# =============================================================================
SIGMAS=${SIGMAS:-"8 10 12 14 16 18"}
EPOCHS=${EPOCHS:-40}
BG=${BG:-3}                       # 3 = eğitimde motorlu + motorsuz slice (val'de her zaman ikisi de var)

module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env
export FLAGELLA_DATA_DIR=/data/horse/ws/beay097h-teamproject_restore/beay097h-teamproject-1790463808/flagellar_motors_data
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"          # proje klasörü = sbatch'i çalıştırdığın klasör
echo "host: $(hostname)  date: $(date)  sigmas: $SIGMAS  epochs: $EPOCHS  bg: $BG"

# ---- 1) veri hazırlama -------------------------------------------------------
if [ -f processed/slices.csv ] && [ -f processed/gt.csv ]; then
  echo "[1] processed/ exists -> skipping data preparation"
else
  python prepare_data.py || { echo "DATA PREPARATION FAILED"; exit 1; }
fi
cat processed/dataset_stats.json

# ---- 2) her sigma: eğitim -> tahmin -> değerlendirme -------------------------
FAILED=""
for s in $SIGMAS; do
  RUN=unet/runs/sigma${s}_bg${BG}_ep${EPOCHS}_seed0
  if [ -f "$RUN/eval_val/metrics.json" ]; then echo "[sigma $s] already done -> skipping"; continue; fi
  echo "=========== sigma $s : train ($(date)) ==========="
  python unet/train_unet.py --sigma "$s" --bg_per_tomo "$BG" --epochs "$EPOCHS" \
    || { echo "TRAIN FAILED sigma $s"; FAILED="$FAILED $s"; continue; }
  echo "=========== sigma $s : predict + evaluate ==========="
  python unet/predict_unet.py --run_dir "$RUN" --checkpoint last \
    && python evaluate.py --run_dir "$RUN" \
    || { echo "EVAL FAILED sigma $s"; FAILED="$FAILED $s"; }
done

# ---- 3) sigma karşılaştırması: tablo + grafikler -----------------------------
if [ -f summarize_results.py ]; then
  python summarize_results.py --runs_dirs unet/runs --out_dir unet/results
else
  echo "summarize_results.py not found -> skipping summary"
fi
[ -n "$FAILED" ] && echo "!!! failed sigmas:$FAILED (re-submit to retry only those)"
echo "done -> unet/results/summary.txt, unet/results/*.png, unet/runs/sigma*/eval_val/summary.txt"
