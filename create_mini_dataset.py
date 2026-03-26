import os
import shutil
import random
from pathlib import Path

def create_mini_dataset(source_base, output_base, train_samples=100, val_samples=20):
    source_base = Path(source_base)
    output_base = Path(output_base)
    
    # Create mini directories
    train_dir = output_base / 'train'
    val_dir = output_base / 'val'
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    
    random.seed(42)
    
    # Process train
    for class_folder in (source_base / 'train').iterdir():
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        (train_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        images = [f for f in class_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        sample_count = min(len(images), train_samples)
        sampled_images = random.sample(images, sample_count)
        
        for img in sampled_images:
            shutil.copy(img, train_dir / class_name / img.name)
            
        print(f"Sampled {sample_count} train images for {class_name}")

    # Process val
    for class_folder in (source_base / 'val').iterdir():
        if not class_folder.is_dir():
            continue
        
        class_name = class_folder.name
        (val_dir / class_name).mkdir(parents=True, exist_ok=True)
        
        images = [f for f in class_folder.iterdir() if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
        sample_count = min(len(images), val_samples)
        sampled_images = random.sample(images, sample_count)
        
        for img in sampled_images:
            shutil.copy(img, val_dir / class_name / img.name)
            
        print(f"Sampled {sample_count} val images for {class_name}")

if __name__ == '__main__':
    print("Creating mini dataset for fast training...")
    create_mini_dataset('dataset', 'mini_dataset')
    print("Mini dataset created at 'mini_dataset/'!")
