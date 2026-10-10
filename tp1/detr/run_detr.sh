#!/bin/bash
#SBATCH --job-name=detr_sanity
#SBATCH --output=detr_sanity_%j.out
#SBATCH --error=detr_sanity_%j.err
#SBATCH --partition=alpha-interactive   # Boş kuyruk
#SBATCH --time=00:20:00                 # Hızlı test için 20 dk
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1               
#SBATCH --mem=16G                       # Transformer modelleri biraz daha RAM sever
#SBATCH --gres=gpu:1

# Senin özel Conda ortamını aktif et
module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env

# Klasörüne git
cd /data/horse/ws/beay097h-teamproject/TeamProject_flagella/detr/

echo "RT-DETR Model Sanity Check basliyor..."
python train_detr.py
echo "Islem tamamlandi! Lutfen .out dosyasini kontrol et."
