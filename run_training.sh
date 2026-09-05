#!/bin/bash
#SBATCH --job-name=flagella_train
#SBATCH --output=train_log_%j.out
#SBATCH --error=train_error_%j.err
#SBATCH --partition=alpha
#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:A100:1
#SBATCH --exclude=i8023


# Activate the Conda environment using the full path
module load Miniconda3/25.5.1-1
source activate /home/beay097h/team-env

# Navigate to your working directory
cd /data/horse/ws/beay097h-teamproject/TeamProject_flagella

#echo "1 & 2. Reading labels and exploring samples..."
#python explore_labels.py
#python prepare_data.py

#echo "3. Splitting dataset into Train and Validation..."
#python split_data.py

echo "4. Testing PyTorch Dataset and generating heatmap..."
python dataset_builder.py

#echo "All data preparation steps completed successfully!"



echo "Step 5: Training the U-Net model..."
python train_model.py

echo "Step 6 & 7: Evaluating the model and performing analysis..."
python evaluate_metrics.py

echo "Process completed successfully! Results are available in the /results folder."

#python  compare_train_val.py
