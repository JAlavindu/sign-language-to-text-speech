import os
from PIL import Image
import random

dataset_path = r"ml-model/datasets/processed/train"
classes = ['A', 'B', 'C']

print(f"Checking image sizes in {dataset_path}...")

for cls in classes:
    cls_path = os.path.join(dataset_path, cls)
    if os.path.exists(cls_path):
        files = os.listdir(cls_path)
        if files:
            # Check 5 random files
            sample_files = random.sample(files, min(len(files), 5))
            print(f"\nClass {cls}:")
            for f in sample_files:
                try:
                    img_path = os.path.join(cls_path, f)
                    with Image.open(img_path) as img:
                        print(f"  {f}: {img.size}")
                except Exception as e:
                    print(f"  Error reading {f}: {e}")
    else:
        print(f"Class {cls} path not found")
