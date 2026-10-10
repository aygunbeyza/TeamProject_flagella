import os
import glob
import torch
from torch.utils.data import DataLoader
from transformers import RTDetrForObjectDetection, RTDetrConfig
from detr_dataset import MotorDETRDataset

# Custom collate function to batch images and target dictionaries (bounding boxes)
def collate_fn(batch):
    pixel_values = [item[0] for item in batch]
    labels = [item[1] for item in batch]
    pixel_values = torch.stack(pixel_values, dim=0)
    return {"pixel_values": pixel_values, "labels": labels}

def main():
    print("1. Preparing dataset and DataLoader...")
    BASE_DIR = "/data/horse/ws/beay097h-teamproject/TeamProject_flagella"
    
    # Find all .png images in the directory
    image_list = glob.glob(os.path.join(BASE_DIR, "**", "*.png"), recursive=True)
    
    if not image_list:
        print("Error: No .png files found!")
        return

    test_image_path = image_list[0]
    test_coord = [256, 256] # Dummy coordinate for testing
    
    # Initialize dataset and DataLoader
    dataset = MotorDETRDataset(image_paths=[test_image_path], coordinates=[test_coord], is_train=True)
    dataloader = DataLoader(dataset, batch_size=1, collate_fn=collate_fn)
    
    # Fetch the first batch
    batch = next(iter(dataloader))
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]
    
    print("2. Downloading/Loading RT-DETR model from Hugging Face...")
    # Configure the model for 1 class (Flagellar Motor)
    config = RTDetrConfig.from_pretrained("PekingU/rtdetr_r50vd")
    config.num_labels = 1
    
    # Load model and ignore size mismatches to replace the pre-trained classification head
    model = RTDetrForObjectDetection.from_pretrained(
        "PekingU/rtdetr_r50vd", 
        config=config, 
        ignore_mismatched_sizes=True
    )
    
    print("3. Performing forward pass...")
    # Pass inputs and targets to the model to compute the Set-Prediction Loss
    outputs = model(pixel_values=pixel_values, labels=labels)
    
    print("\n" + "="*50)
    print("SUCCESS! MODEL AWAKE AND FUNCTIONAL!")
    print(f"Calculated Set-Prediction Loss: {outputs.loss.item()}")
    print("="*50 + "\n")

if __name__ == "__main__":
    main()
