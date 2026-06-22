#!/bin/bash
#SBATCH --job-name=flagella_data_prep
#SBATCH --output=flagella_log_%j.out
#SBATCH --error=flagella_error_%j.err
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1

# Load Conda and activate the environment
module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env

# Navigate to your working directory
cd /data/horse/ws/beay097h-teamproject/TeamProject_flagella

echo "1 & 2. Reading labels and exploring samples..."
python explore_labels.py
python prepare_data.py

echo "3. Splitting dataset into Train and Validation..."
python split_data.py

echo "4. Testing PyTorch Dataset and generating heatmap..."
python dataset_builder.py

echo "All data preparation steps completed successfully!"
