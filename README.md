# TeamProject: Locating Bacterial Flagellar Motors (2025)

This project aims to detect bacterial flagellar motors in 3D cryo-electron tomograms (cryo-EM) using Deep Learning. The code is optimized to run on a High-Performance Computing (HPC) Cluster using Slurm.

## 📂 Project Structure

* **`requirements.txt`**: Required Python libraries (numpy, pandas, torch, etc.).
* **`explore_labels.py`**: Reads `train_labels.csv` to check motor coordinates and tomogram statistics.
* **`prepare_data.py`**: Loads a 2D slice, normalizes it, marks the motor location, and saves a check image.
* **`split_data.py`**: Splits the data into Train (80%) and Validation (20%) based on `tomo_id` to prevent data leakage.
* **`dataset_builder.py`**: Contains the PyTorch `MotorSliceDataset`. It generates 2D normalized slices and 2D Gaussian heatmaps as targets for the model.
* **`train_model.py`**: Defines the 2D U-Net architecture and runs the training loop. It saves the trained weights (`unet_model.pth`) and learning curves.
* **`evaluate_metrics.py`**: Contains the peak detection logic (Non-Maximum Suppression) and evaluates the model's Precision, Recall, and F1-score on the validation set.
* **`run_data.sh`**: The Slurm script to run data preparation on the compute nodes.
* **`run_training.sh`**: The Slurm script requesting GPU resources to execute the model training and evaluation.
* **`results/`**: Folder for all saved sanity check images, training loss curves, prediction samples, and PR-metric plots.
* **`train_ids.txt` & `val_ids.txt`**: Saved lists of tomogram IDs for training and validation.

## 📌 Project Steps Explained

* **Step 1 & 2: Understand Data & Read Labels**
  We look at the 3D images (tomograms) and read the `train_labels.csv` file. This file tells us exactly where the motors are using x, y, and z coordinates.
  *Scripts used:* `explore_labels.py` and `prepare_data.py`

* **Step 3: Split into Train and Validation**
  We divide our tomograms into two groups. "Train" data is used to teach the AI model. "Validation" data is kept secret to test the model later. We split them carefully so the model doesn't cheat by memorizing slices.
  *Script used:* `split_data.py`

* **Step 4: Build a 2D Dataset**
  Full 3D data is very large, so we start simple. We extract 2D image slices that contain a motor. Then, we create a "heatmap" (a target image with a bright spot exactly where the motor is). The AI will look at the slice and learn to generate this bright spot.
  *Script used:* `dataset_builder.py`

* **Step 5: Train a 2D U-Net**
  We train an image-to-image neural network (U-Net) using the data from Step 4. The model looks at a normalized slice and tries to regress the Gaussian heatmap. It learns over multiple epochs by minimizing the Mean Squared Error (MSE) loss, saving the best brain (`unet_model.pth`).
  *Script used:* `train_model.py`

* **Step 6: Turn Heatmaps into Detections**
  The model outputs probability heatmaps, but we need exact (x, y) coordinates. We use a peak detection function with Non-Maximum Suppression (NMS) to find the brightest pixels (local maxima) and filter out duplicate detections that are too close to each other.
  *Script used:* Integrated into `evaluate_metrics.py`

* **Step 7: Evaluate Model Performance**
  We test our trained model on the unseen validation set. By comparing the predicted motor coordinates against the actual ground truth, we calculate Precision (accuracy of detections) and Recall (how many real motors we found). We sweep across different confidence thresholds to plot a PR Curve and find the optimal F1-score.
  *Script used:* `evaluate_metrics.py`

## 🚀 How to Run

**1. Setup Environment:**
Activate your Conda environment and install the required libraries:
```bash
source activate /home/beay097h/team-env
pip install -r requirements.txt
