"""
Dataset Preparation Script
Merges datasets, splits into train/val/test, and creates organized structure
"""

import os
import shutil
import glob
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
SIGNALAPHASET_PATH = os.getenv("SIGNALAPHASET_PATH", r"e:\UNI sub\ICT\3rd yr\HCI\SignAlphaSet")
ASL_DATASET_PATH = os.getenv("ASL_DATASET_PATH", r"e:\UNI sub\ICT\3rd yr\HCI\asl_dataset")
OUTPUT_PATH = os.getenv("PROCESSED_DATASET_PATH", r"e:\UNI sub\ICT\3rd yr\HCI\sign-language-glove\ml-model\datasets\processed")

# If paths are relative, make them absolute based on project root
if not os.path.isabs(OUTPUT_PATH):
    OUTPUT_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), OUTPUT_PATH)

# Split ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Random seed for reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def get_all_images(dataset_path, class_folders):
    """Get all image paths organized by class"""
    data = []
    
    for class_name in tqdm(class_folders, desc=f"Scanning {Path(dataset_path).name}"):
        class_path = os.path.join(dataset_path, class_name)
        if not os.path.isdir(class_path):
            continue
        
        # Get all image files
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        images = []
        for ext in image_extensions:
            images.extend(glob.glob(os.path.join(class_path, ext)))
        
        for img_path in images:
            # Normalize class name (uppercase)
            normalized_class = class_name.upper()
            data.append({
                'path': img_path,
                'class': normalized_class,
                'source': Path(dataset_path).name
            })
    
    return data

def create_splits(data_df, train_ratio, val_ratio, test_ratio):
    """Split data into train/validation/test sets"""
    print(f"\n{'='*60}")
    print("Creating Train/Val/Test Splits")
    print(f"{'='*60}")
    
    # Group by class to ensure stratified split
    train_data = []
    val_data = []
    test_data = []
    
    for class_name in data_df['class'].unique():
        class_data = data_df[data_df['class'] == class_name]
        
        # First split: train vs (val + test)
        train_class, temp_class = train_test_split(
            class_data, 
            test_size=(val_ratio + test_ratio),
            random_state=RANDOM_SEED
        )
        
        # Second split: val vs test
        val_class, test_class = train_test_split(
            temp_class,
            test_size=(test_ratio / (val_ratio + test_ratio)),
            random_state=RANDOM_SEED
        )
        
        train_data.append(train_class)
        val_data.append(val_class)
        test_data.append(test_class)
    
    train_df = pd.concat(train_data, ignore_index=True)
    val_df = pd.concat(val_data, ignore_index=True)
    test_df = pd.concat(test_data, ignore_index=True)
    
    print(f"\nSplit Summary:")
    print(f"  Training:   {len(train_df):>6} images ({len(train_df)/len(data_df)*100:.1f}%)")
    print(f"  Validation: {len(val_df):>6} images ({len(val_df)/len(data_df)*100:.1f}%)")
    print(f"  Test:       {len(test_df):>6} images ({len(test_df)/len(data_df)*100:.1f}%)")
    print(f"  Total:      {len(data_df):>6} images")
    
    return train_df, val_df, test_df

def copy_images_to_split(df, split_name, output_path):
    """Copy images to organized folder structure"""
    split_path = os.path.join(output_path, split_name)
    print(f"\nCopying images to {split_name} folder...")
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Copying {split_name}"):
        class_folder = os.path.join(split_path, row['class'])
        os.makedirs(class_folder, exist_ok=True)
        
        # Generate new filename to avoid conflicts
        source_file = Path(row['path'])
        new_filename = f"{row['source']}_{source_file.stem}{source_file.suffix}"
        dest_path = os.path.join(class_folder, new_filename)
        
        try:
            shutil.copy2(row['path'], dest_path)
        except Exception as e:
            print(f"Error copying {row['path']}: {e}")
    
    print(f"✓ {split_name} complete: {len(df)} images copied")

def save_metadata(train_df, val_df, test_df, output_path):
    """Save metadata CSV files"""
    print(f"\n{'='*60}")
    print("Saving Metadata")
    print(f"{'='*60}")
    
    # Add split column
    train_df['split'] = 'train'
    val_df['split'] = 'validation'
    test_df['split'] = 'test'
    
    # Combine all
    all_data = pd.concat([train_df, val_df, test_df], ignore_index=True)
    
    # Save individual splits
    train_df.to_csv(os.path.join(output_path, 'train_metadata.csv'), index=False)
    val_df.to_csv(os.path.join(output_path, 'val_metadata.csv'), index=False)
    test_df.to_csv(os.path.join(output_path, 'test_metadata.csv'), index=False)
    
    # Save combined
    all_data.to_csv(os.path.join(output_path, 'all_metadata.csv'), index=False)
    
    print(f"✓ Metadata saved to {output_path}")
    
    # Create class mapping
    classes = sorted(all_data['class'].unique())
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    idx_to_class = {idx: cls for cls, idx in class_to_idx.items()}
    
    # Save class mapping
    with open(os.path.join(output_path, 'class_mapping.json'), 'w') as f:
        json.dump({
            'class_to_idx': class_to_idx,
            'idx_to_class': idx_to_class,
            'num_classes': len(classes),
            'classes': classes
        }, f, indent=2)
    
    print(f"✓ Class mapping saved: {len(classes)} classes")
    
    return all_data, class_to_idx

def print_split_statistics(train_df, val_df, test_df):
    """Print detailed statistics about splits"""
    print(f"\n{'='*60}")
    print("Split Statistics by Class")
    print(f"{'='*60}")
    
    classes = sorted(train_df['class'].unique())
    
    print(f"\n{'Class':<8} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 60)
    
    for cls in classes:
        train_count = len(train_df[train_df['class'] == cls])
        val_count = len(val_df[val_df['class'] == cls])
        test_count = len(test_df[test_df['class'] == cls])
        total = train_count + val_count + test_count
        
        print(f"{cls:<8} {train_count:<10} {val_count:<10} {test_count:<10} {total:<10}")
    
    print("-" * 60)
    print(f"{'TOTAL':<8} {len(train_df):<10} {len(val_df):<10} {len(test_df):<10} {len(train_df)+len(val_df)+len(test_df):<10}")

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("ASL DATASET PREPARATION")
    print("="*60)
    
    # Create output directory
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    # Collect all data
    all_data = []
    
    # Process SignAlphaSet (A-Z)
    if os.path.exists(SIGNALAPHASET_PATH):
        print(f"\nProcessing SignAlphaSet...")
        alpha_folders = [chr(i) for i in range(ord('A'), ord('Z')+1)]
        alpha_data = get_all_images(SIGNALAPHASET_PATH, alpha_folders)
        all_data.extend(alpha_data)
        print(f"✓ Found {len(alpha_data)} images from SignAlphaSet")
    
    # Process asl_dataset (a-z, 0-9)
    if os.path.exists(ASL_DATASET_PATH):
        print(f"\nProcessing asl_dataset...")
        asl_folders = [chr(i) for i in range(ord('a'), ord('z')+1)] + \
                      [str(i) for i in range(10)]
        asl_data = get_all_images(ASL_DATASET_PATH, asl_folders)
        all_data.extend(asl_data)
        print(f"✓ Found {len(asl_data)} images from asl_dataset")
    
    if not all_data:
        print("\nError: No images found! Check dataset paths.")
        return
    
    # Create DataFrame
    data_df = pd.DataFrame(all_data)
    
    print(f"\n{'='*60}")
    print(f"Total images collected: {len(data_df)}")
    print(f"Total classes: {data_df['class'].nunique()}")
    print(f"Classes: {sorted(data_df['class'].unique())}")
    print(f"{'='*60}")
    
    # Create splits
    train_df, val_df, test_df = create_splits(data_df, TRAIN_RATIO, VAL_RATIO, TEST_RATIO)
    
    # Print detailed statistics
    print_split_statistics(train_df, val_df, test_df)
    
    # Copy images to organized structure
    print(f"\n{'='*60}")
    print("Organizing Files")
    print(f"{'='*60}")
    
    copy_images_to_split(train_df, 'train', OUTPUT_PATH)
    copy_images_to_split(val_df, 'validation', OUTPUT_PATH)
    copy_images_to_split(test_df, 'test', OUTPUT_PATH)
    
    # Save metadata
    all_data_df, class_mapping = save_metadata(train_df, val_df, test_df, OUTPUT_PATH)
    
    # Final summary
    print(f"\n{'='*60}")
    print("PREPARATION COMPLETE!")
    print(f"{'='*60}")
    print(f"\nDataset location: {OUTPUT_PATH}")
    print(f"\nFolder structure:")
    print(f"  {OUTPUT_PATH}/")
    print(f"    ├── train/          ({len(train_df)} images)")
    print(f"    ├── validation/     ({len(val_df)} images)")
    print(f"    ├── test/           ({len(test_df)} images)")
    print(f"    ├── train_metadata.csv")
    print(f"    ├── val_metadata.csv")
    print(f"    ├── test_metadata.csv")
    print(f"    ├── all_metadata.csv")
    print(f"    └── class_mapping.json")
    print(f"\n✓ Ready for training!\n")

if __name__ == "__main__":
    main()
