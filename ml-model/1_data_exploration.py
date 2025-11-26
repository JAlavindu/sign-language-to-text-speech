"""
Data Exploration Script
Analyzes the ASL datasets and generates statistics
"""

import os
import glob
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Dataset paths
SIGNALAPHASET_PATH = os.getenv("SIGNALAPHASET_PATH", r"e:\UNI sub\ICT\3rd yr\HCI\SignAlphaSet")
ASL_DATASET_PATH = os.getenv("ASL_DATASET_PATH", r"e:\UNI sub\ICT\3rd yr\HCI\asl_dataset")

def count_images_per_class(dataset_path, dataset_name):
    """Count images in each class folder"""
    print(f"\n{'='*60}")
    print(f"Analyzing: {dataset_name}")
    print(f"{'='*60}")
    
    class_counts = {}
    folders = sorted([f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f))])
    
    for folder in folders:
        folder_path = os.path.join(dataset_path, folder)
        images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                 glob.glob(os.path.join(folder_path, "*.jpeg")) + \
                 glob.glob(os.path.join(folder_path, "*.png"))
        class_counts[folder] = len(images)
    
    # Print statistics
    total_images = sum(class_counts.values())
    print(f"\nTotal classes: {len(class_counts)}")
    print(f"Total images: {total_images}")
    print(f"Average images per class: {total_images / len(class_counts):.0f}")
    print(f"Min images in a class: {min(class_counts.values())} ({min(class_counts, key=class_counts.get)})")
    print(f"Max images in a class: {max(class_counts.values())} ({max(class_counts, key=class_counts.get)})")
    
    # Show distribution
    print(f"\n{'Class':<10} {'Count':<10} {'Bar'}")
    print("-" * 50)
    max_count = max(class_counts.values())
    for cls, count in sorted(class_counts.items()):
        bar_length = int((count / max_count) * 30)
        bar = "█" * bar_length
        print(f"{cls:<10} {count:<10} {bar}")
    
    return class_counts

def check_image_quality(dataset_path, sample_size=10):
    """Check a sample of images for quality issues"""
    print(f"\n{'='*60}")
    print("Checking Image Quality")
    print(f"{'='*60}")
    
    corrupted = []
    sizes = []
    
    # Get random sample of images
    all_images = []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_images.append(os.path.join(root, file))
    
    sample_images = np.random.choice(all_images, min(sample_size, len(all_images)), replace=False)
    
    for img_path in sample_images:
        try:
            img = Image.open(img_path)
            sizes.append(img.size)
            img.verify()  # Check if image is corrupted
        except Exception as e:
            corrupted.append((img_path, str(e)))
    
    print(f"\nSampled {sample_size} images")
    print(f"Corrupted images found: {len(corrupted)}")
    
    if corrupted:
        print("\nCorrupted images:")
        for img_path, error in corrupted:
            print(f"  - {img_path}: {error}")
    
    if sizes:
        unique_sizes = Counter(sizes)
        print(f"\nImage dimensions found:")
        for size, count in unique_sizes.most_common(5):
            print(f"  {size[0]}x{size[1]}: {count} images")
    
    return corrupted, sizes

def visualize_samples(dataset_path, dataset_name, num_classes=5, samples_per_class=3):
    """Visualize random samples from dataset"""
    print(f"\n{'='*60}")
    print(f"Visualizing Samples from {dataset_name}")
    print(f"{'='*60}")
    
    folders = sorted([f for f in os.listdir(dataset_path) 
                     if os.path.isdir(os.path.join(dataset_path, f)) 
                     and f != 'asl_dataset'])[:num_classes]
    
    fig, axes = plt.subplots(num_classes, samples_per_class, figsize=(12, 3*num_classes))
    fig.suptitle(f'{dataset_name} - Sample Images', fontsize=16)
    
    for i, folder in enumerate(folders):
        folder_path = os.path.join(dataset_path, folder)
        images = glob.glob(os.path.join(folder_path, "*.jpg")) + \
                 glob.glob(os.path.join(folder_path, "*.jpeg"))
        
        if len(images) < samples_per_class:
            samples = images
        else:
            samples = np.random.choice(images, samples_per_class, replace=False)
        
        for j, img_path in enumerate(samples):
            try:
                img = Image.open(img_path)
                if num_classes == 1:
                    ax = axes[j]
                else:
                    ax = axes[i, j]
                ax.imshow(img)
                ax.axis('off')
                if j == 0:
                    ax.set_title(f'Class: {folder}', fontsize=12, fontweight='bold')
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
    
    plt.tight_layout()
    output_path = os.path.join("ml-model", "reports", f"{dataset_name}_samples.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSample visualization saved to: {output_path}")
    plt.close()

def plot_class_distribution(class_counts, dataset_name):
    """Plot class distribution"""
    plt.figure(figsize=(16, 6))
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    plt.bar(classes, counts, color='steelblue', alpha=0.8, edgecolor='black')
    plt.xlabel('Class', fontsize=12)
    plt.ylabel('Number of Images', fontsize=12)
    plt.title(f'{dataset_name} - Class Distribution', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    
    # Add count labels on bars
    for i, (cls, count) in enumerate(zip(classes, counts)):
        plt.text(i, count + max(counts)*0.01, str(count), 
                ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = os.path.join("ml-model", "reports", f"{dataset_name}_distribution.png")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Distribution plot saved to: {output_path}")
    plt.close()

def main():
    """Main execution"""
    print("\n" + "="*60)
    print("ASL DATASET EXPLORATION")
    print("="*60)
    
    # Create reports directory
    os.makedirs("ml-model/reports", exist_ok=True)
    
    # Analyze SignAlphaSet
    if os.path.exists(SIGNALAPHASET_PATH):
        counts_alpha = count_images_per_class(SIGNALAPHASET_PATH, "SignAlphaSet")
        plot_class_distribution(counts_alpha, "SignAlphaSet")
        visualize_samples(SIGNALAPHASET_PATH, "SignAlphaSet", num_classes=5, samples_per_class=4)
        check_image_quality(SIGNALAPHASET_PATH, sample_size=20)
    else:
        print(f"SignAlphaSet not found at: {SIGNALAPHASET_PATH}")
    
    # Analyze asl_dataset
    if os.path.exists(ASL_DATASET_PATH):
        counts_asl = count_images_per_class(ASL_DATASET_PATH, "asl_dataset")
        plot_class_distribution(counts_asl, "asl_dataset")
        visualize_samples(ASL_DATASET_PATH, "asl_dataset", num_classes=5, samples_per_class=4)
        check_image_quality(ASL_DATASET_PATH, sample_size=20)
    else:
        print(f"asl_dataset not found at: {ASL_DATASET_PATH}")
    
    # Combined summary
    print(f"\n{'='*60}")
    print("COMBINED DATASET SUMMARY")
    print(f"{'='*60}")
    if os.path.exists(SIGNALAPHASET_PATH) and os.path.exists(ASL_DATASET_PATH):
        total_images = sum(counts_alpha.values()) + sum(counts_asl.values())
        total_classes = len(set(list(counts_alpha.keys()) + list(counts_asl.keys())))
        print(f"Total unique classes: {total_classes}")
        print(f"Total images: {total_images}")
        print(f"Average images per class: {total_images / total_classes:.0f}")
    
    print(f"\n{'='*60}")
    print("Exploration complete! Check ml-model/reports/ for visualizations.")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
