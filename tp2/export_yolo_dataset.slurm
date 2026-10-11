#!/bin/bash
#SBATCH --job-name=export_yolo
#SBATCH --output=yolo/logs/export_%j.out
#SBATCH --error=yolo/logs/export_%j.err
#SBATCH --partition=alpha
#SBATCH --time=01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023
# YOLO / RT-DETR veri seti dönüştürme (bir kez çalıştır): sbatch export_yolo_dataset.slurm
# Girdi : processed/slices.csv, gt.csv     Çıktı: processed/ultralytics_dataset/

module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env
export FLAGELLA_DATA_DIR=/data/horse/ws/beay097h-teamproject_restore/beay097h-teamproject-1790463808/flagellar_motors_data
export PYTHONUNBUFFERED=1
cd "$SLURM_SUBMIT_DIR"

python common/export_yolo_dataset.py
