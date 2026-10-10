#!/bin/bash
#SBATCH --job-name=gpu_test
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --output=gpu_test_%j.out
#SBATCH --error=gpu_test_%j.err

echo "START"
hostname
nvidia-smi
echo "END"
