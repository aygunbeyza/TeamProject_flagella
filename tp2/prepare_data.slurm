#!/bin/bash
#SBATCH --job-name=prepare_data
#SBATCH --output=unet/logs/prepare_data_%j.out
#SBATCH --error=unet/logs/prepare_data_%j.err
#SBATCH --partition=alpha
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023

module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env

export FLAGELLA_DATA_DIR=/data/horse/ws/beay097h-teamproject_restore/beay097h-teamproject-1790463808/flagellar_motors_data
cd "$SLURM_SUBMIT_DIR"

python prepare_data.py
