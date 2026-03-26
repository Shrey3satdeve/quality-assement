import os
import shutil
from pathlib import Path
import random

def prepare_yolo_cls_dataset(source_base, output_base, split_ratio=0.8):
    source_base = Path(source_base)
    output_base = Path(output_base)
    
    # Create output directories
    train_dir = output_base / 'train'
    val_dir = output_base / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
         
    categories = ['Bad Quality_Fruits', 'Good Quality_Fruits', 'Mixed_Quality_Fruits']
    
    random.seed(42)
    for category in categories:
        cat_path = source_base / category
        if not cat_path.exists():
            continue
            
        for class_folder in cat_path.iterdir():
            if not class_folder.is_dir() or class_folder.name == '__pycache__':
                continue
                
            class_name = class_folder.name
            (train_dir / class_name).mkdir(parents=True, exist_ok=True)
            (val_dir / class_name).mkdir(parents=True, exist_ok=True)
            
            images = [f for f in class_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            if len(images) == 0:
                continue
                
            random.shuffle(images)
            split_idx = int(len(images) * split_ratio)
            train_imgs = images[:split_idx]
            val_imgs = images[split_idx:]
            
            for img in train_imgs:
                shutil.copy(img, train_dir / class_name / img.name)
            for img in val_imgs:
                shutil.copy(img, val_dir / class_name / img.name)
                
            print(f"Processed {class_name}: {len(train_imgs)} train, {len(val_imgs)} val")

if __name__ == '__main__':
    print("Preparing YOLO classification dataset...")
    prepare_yolo_cls_dataset('data', 'dataset')
    print("Dataset prepared at 'dataset/' folder!")