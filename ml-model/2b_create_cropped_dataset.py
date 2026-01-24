"""
Dataset Cropping Script
-----------------------
This script processes the existing 'processed' dataset and creates a new 'cropped_dataset'.
It runs MediaPipe Hands on every training image to crop ONLY the hand.
This ensures the training data matches the real-time camera input.
"""

import os
import cv2
import mediapipe as mp
import glob
from pathlib import Path
from tqdm import tqdm
import shutil

# Configuration
INPUT_DIR = os.path.join(os.path.dirname(__file__), "datasets", "processed")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "datasets", "cropped")
PADDING = 20

def init_mediapipe():
    mp_hands = mp.solutions.hands
    # Static image mode is crucial for processing a dataset of independent images
    hands = mp_hands.Hands(
        static_image_mode=True, 
        max_num_hands=1, 
        min_detection_confidence=0.5
    )
    return hands

def get_square_bbox(h_img, w_img, landmarks, padding=PADDING):
    """Calculate a square bounding box around the hand landmarks"""
    x_min, y_min = w_img, h_img
    x_max, y_max = 0, 0

    for lm in landmarks.landmark:
        x, y = int(lm.x * w_img), int(lm.y * h_img)
        x_min = min(x_min, x)
        y_min = min(y_min, y)
        x_max = max(x_max, x)
        y_max = max(y_max, y)

    w_box = x_max - x_min
    h_box = y_max - y_min
    
    # Calculate center
    center_x = x_min + w_box // 2
    center_y = y_min + h_box // 2
    
    # Make square
    max_dim = max(w_box, h_box) + (padding * 2)
    half_dim = max_dim // 2
    
    # Coordinates
    x1 = max(center_x - half_dim, 0)
    y1 = max(center_y - half_dim, 0)
    x2 = min(center_x + half_dim, w_img)
    y2 = min(center_y + half_dim, h_img)
    
    return x1, y1, x2, y2

def process_dataset():
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory not found: {INPUT_DIR}")
        print("Please run '2_prepare_dataset.py' first.")
        return

    print(f"Processing images from: {INPUT_DIR}")
    print(f"Saving crops to:      {OUTPUT_DIR}")
    
    hands = init_mediapipe()
    
    # Walk through train, validation, test
    for split in ['train', 'validation', 'test']:
        split_path = os.path.join(INPUT_DIR, split)
        if not os.path.exists(split_path):
            continue
            
        print(f"\nProcessing {split} set...")
        
        # Get all class folders
        class_folders = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
        
        for class_name in tqdm(class_folders, desc=f"Classes in {split}"):
            src_class_dir = os.path.join(split_path, class_name)
            dst_class_dir = os.path.join(OUTPUT_DIR, split, class_name)
            
            os.makedirs(dst_class_dir, exist_ok=True)
            
            # Process images
            images = glob.glob(os.path.join(src_class_dir, "*.*"))
            
            for img_path in images:
                filename = os.path.basename(img_path)
                dst_path = os.path.join(dst_class_dir, filename)
                
                # Check if it's an image
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    continue
                
                try:
                    # Read image
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                        
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    h, w, _ = img.shape
                    
                    # Detect hand
                    results = hands.process(img_rgb)
                    
                    if results.multi_hand_landmarks:
                        # Use the first hand found
                        landmarks = results.multi_hand_landmarks[0]
                        x1, y1, x2, y2 = get_square_bbox(h, w, landmarks)
                        
                        # Crop
                        crop = img[y1:y2, x1:x2]
                        if crop.size > 0:
                            # Resize to unified size to save space/time later (optional but good idea)
                            crop = cv2.resize(crop, (224, 224))
                            cv2.imwrite(dst_path, crop)
                    else:
                        # Fallback: If no hand detected, use the center crop or original?
                        # Using original might re-introduce bad data, but omitting it reduces dataset size.
                        # Strategy: Resize original to 224x224 and save.
                        # This handles cases where the hand is the main subject already.
                        resized = cv2.resize(img, (224, 224))
                        cv2.imwrite(dst_path, resized)
                        
                except Exception as e:
                    # print(f"Error processing {filename}: {e}")
                    pass

    # Copy metadata files
    for meta in ['class_mapping.json', 'train_metadata.csv', 'val_metadata.csv', 'test_metadata.csv']:
        src = os.path.join(INPUT_DIR, meta)
        dst = os.path.join(OUTPUT_DIR, meta)
        if os.path.exists(src):
            shutil.copy2(src, dst)

    print("\n✓ Cropping complete!")
    print(f"New dataset is ready at: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_dataset()
