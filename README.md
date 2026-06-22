# TeamProject: Locating Bacterial Flagellar Motors (2025)

This project aims to detect bacterial flagellar motors in 3D cryo-electron tomograms (cryo-EM) using Deep Learning. The code is optimized to run on a High-Performance Computing (HPC) Cluster using Slurm.

## 📂 Project Structure

* **`requirements.txt`**: Required Python libraries (numpy, pandas, torch, etc.).
* **`explore_labels.py`**: Reads `train_labels.csv` to check motor coordinates and tomogram statistics.
* **`prepare_data.py`**: Loads a 2D slice, normalizes it, marks the motor location, and saves a check image.
* **`split_data.py`**: Splits the data into Train (80%) and Validation (20%) based on `tomo_id` to prevent data leakage.
* **`dataset_builder.py`**: Contains the PyTorch `MotorSliceDataset`. It generates 2D normalized slices and 2D Gaussian heatmaps as targets for the model.
* **`run_data.sh`**: The Slurm script to run data preparation on the compute nodes.
* **`figures/`**: Folder for all saved sanity check images and heatmaps.
* **`train_ids.txt` & `val_ids.txt`**: Saved lists of tomogram IDs for training and validation.

## 📌 Project Steps Explained (Completed So Far)

* **Step 1 & 2: Understand Data & Read Labels**
  We look at the 3D images (tomograms) and read the `train_labels.csv` file. This file tells us exactly where the motors are using x, y, and z coordinates.
  *Scripts used:* `explore_labels.py` and `prepare_data.py`

* **Step 3: Split into Train and Validation**
  We divide our tomograms into two groups. "Train" data is used to teach the AI model. "Validation" data is kept secret to test the model later. We split them carefully so the model doesn't cheat by memorizing slices.
  *Script used:* `split_data.py`

* **Step 4: Build a 2D Dataset**
  Full 3D data is very large, so we start simple. We extract 2D image slices that contain a motor. Then, we create a "heatmap" (a target image with a bright spot exactly where the motor is). The AI will look at the slice and learn to generate this bright spot.
  *Script used:* `dataset_builder.py`

## 🚀 How to Run

**1. Setup Environment:**
Activate your Conda environment and install the required libraries:
```bash
source activate team-env
pip install -r requirements.txt
