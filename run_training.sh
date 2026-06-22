#!/bin/bash
#SBATCH --job-name=flagella_train_eval
#SBATCH --output=train_log_%j.out
#SBATCH --error=train_error_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

# Activate the Conda environment using the full path
source activate /tmp/conda_envs_beay097h/team-env

# Navigate to your working directory
cd /data/horse/ws/beay097h-teamproject/TeamProject_flagella

echo "Step 5: Training the U-Net model..."
python train_model.py

echo "Step 6 & 7: Evaluating the model and performing analysis..."
python evaluate_metrics.py

echo "Process completed successfully! Results are available in the /results folder."
