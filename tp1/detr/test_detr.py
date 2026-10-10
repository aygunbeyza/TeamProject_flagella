import os
import glob
import torch
from detr_dataset import MotorDETRDataset 

# Senin ana proje dizinin
BASE_DIR = "/data/horse/ws/beay097h-teamproject/TeamProject_flagella"

# Klasördeki (.png) uzantılı ilk resmi otomatik olarak arayıp bulalım
image_list = glob.glob(os.path.join(BASE_DIR, "**", "*.png"), recursive=True)

if not image_list:
    print("Hata: Proje klasöründe hiç .png uzantılı resim bulunamadı! Uzantı .tiff veya .mrc ise kodu güncellemeliyiz.")
else:
    test_image_path = image_list[0] # Listeden ilk resmi seç
    test_coord = [256, 256] # Test için örnek bir motor merkez koordinatı

    print(f"Otomatik bulunan resim test ediliyor: {test_image_path}")

    # Veri setini başlat (1 resimlik)
    dataset = MotorDETRDataset(image_paths=[test_image_path], coordinates=[test_coord], is_train=True)

    # Resmi ve hedef sözlüğünü çek
    img, target = dataset[0]

    print("\n" + "="*30)
    print("--- RT-DETR VERİ SETİ TEST SONUCU ---")
    print("="*30)
    print(f"Resim Boyutu (Tensor): {img.shape}")
    print(f"Kutu (Box) Koordinatları: {target['boxes']}")
    print(f"Sınıf Etiketi: {target['labels']}")
    print("="*30 + "\n")
