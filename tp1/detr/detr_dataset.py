import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class MotorDETRDataset(Dataset):
    """
    Dataset builder for RT-DETR object detection using FULL IMAGES.
    No cropping is applied.
    """
    def __init__(self, image_paths, coordinates, is_train=True):
        self.image_paths = image_paths
        self.coordinates = coordinates
        self.is_train = is_train
        
        # Motorun tahmini boyutu (Piksel cinsinden). Kendi verine göre güncelle!
        self.box_size = 40.0 

        # Albumentations pipeline: Kırpma (Crop) YOK, tüm resim kullanılıyor.
        if self.is_train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
        else:
            self.transform = A.Compose([
            A.PadIfNeeded(pad_height_divisor=32, pad_width_divisor=32),
                ToTensorV2()
            ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 1. Ham tomogram kesitini yükle
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        image = np.array(image) 

        # 2. Bu resimdeki motorun merkez koordinatlarını al
        x_center, y_center = self.coordinates[idx]

        # 3. Kutu (Bounding Box) oranlarını hesapla (TÜM resme göre)
        img_height, img_width = image.shape[:2]
        norm_x = x_center / img_width
        norm_y = y_center / img_height
        norm_w = self.box_size / img_width
        norm_h = self.box_size / img_height
        
        bboxes = [[norm_x, norm_y, norm_w, norm_h]]
        class_labels = [1] # 1 = Motor Sınıfı

        # 4. Augmentation uygula (Kırpma olmadan sadece döndürmeler)
        transformed = self.transform(image=image, bboxes=bboxes, class_labels=class_labels)
        
        # 5. Dönüştürülmüş verileri çıkar ve Tensor'e çevir
        img_tensor = transformed['image'] / 255.0 
        aug_bboxes = transformed['bboxes']
        aug_labels = transformed['class_labels']

        # 6. RT-DETR için hedef (Target) sözlüğünü oluştur
        target = {}
        if len(aug_bboxes) > 0:
            target['boxes'] = torch.tensor(aug_bboxes, dtype=torch.float32)
            target['labels'] = torch.tensor(aug_labels, dtype=torch.int64)
        else:
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)

        return img_tensor, target
