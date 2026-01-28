"""
Main Training Script for ASL Recognition Model
Uses Transfer Learning with MobileNetV2 (PyTorch)
"""

import os
import json
import time
import copy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from dotenv import load_dotenv

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

# Load environment variables
load_dotenv()

# Configuration
# Get paths from env or use defaults
# CHANGED: Using processed dataset directly to include new data without waiting for cropping
DATASET_PATH = os.getenv("PROCESSED_DATASET_PATH", r"E:\Lavindu\HCI\sign-language-to-text-speech\ml-model\datasets\processed")
MODEL_SAVE_PATH = os.getenv("MODEL_SAVE_PATH", r"E:\Lavindu\HCI\sign-language-to-text-speech\ml-model\models")
LOGS_PATH = os.getenv("LOGS_PATH", r"E:\Lavindu\HCI\sign-language-to-text-speech\ml-model\logs")

# Resolve relative paths
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if not os.path.isabs(DATASET_PATH):
    DATASET_PATH = os.path.join(project_root, DATASET_PATH)
if not os.path.isabs(MODEL_SAVE_PATH):
    MODEL_SAVE_PATH = os.path.join(project_root, MODEL_SAVE_PATH)
if not os.path.isabs(LOGS_PATH):
    LOGS_PATH = os.path.join(project_root, LOGS_PATH)

# Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# Create directories
os.makedirs(MODEL_SAVE_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

def load_class_mapping():
    """Load class names and mapping"""
    mapping_path = os.path.join(DATASET_PATH, 'class_mapping.json')
    if not os.path.exists(mapping_path):
        raise FileNotFoundError(f"Mapping file not found at {mapping_path}")
        
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    return mapping

def create_dataloaders():
    """Create PyTorch DataLoaders with augmentation"""
    print(f"\n{'='*60}")
    print("Creating Data Loaders")
    print(f"{'='*60}")
    
    # Data augmentation and normalization for training
    # Just normalization for validation/test
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomRotation(20),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=15, scale=(0.8, 1.2)),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'validation': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(DATASET_PATH, x), data_transforms[x])
                      for x in ['train', 'validation', 'test']}
    
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                                                 shuffle=(x == 'train'), num_workers=0)
                   for x in ['train', 'validation', 'test']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'validation', 'test']}
    class_names = image_datasets['train'].classes
    
    print(f"✓ Training samples: {dataset_sizes['train']}")
    print(f"✓ Validation samples: {dataset_sizes['validation']}")
    print(f"✓ Test samples: {dataset_sizes['test']}")
    print(f"✓ Number of classes: {len(class_names)}")
    print(f"✓ Batch size: {BATCH_SIZE}")
    
    return dataloaders, dataset_sizes, class_names

def train_model(model, criterion, optimizer, scheduler, dataloaders, dataset_sizes, device, num_epochs=25):
    """Training loop"""
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'validation']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
            
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')
            
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.item())
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc.item())

            # deep copy the model
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), os.path.join(MODEL_SAVE_PATH, 'best_model.pth'))

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, history

def plot_history(history, model_name):
    """Plot training history"""
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_acc'], label='Train Accuracy')
    plt.plot(history['val_acc'], label='Val Accuracy')
    plt.title('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title('Loss')
    plt.legend()
    
    plot_path = os.path.join(LOGS_PATH, f'{model_name}_history.png')
    plt.savefig(plot_path)
    print(f"Training history saved to {plot_path}")

def main():
    print("\n" + "="*60)
    print("ASL SIGN LANGUAGE RECOGNITION - PyTorch Training")
    print("="*60)
    
    # Check for GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load class mapping
    class_mapping = load_class_mapping()
    num_classes = class_mapping['num_classes']
    print(f"✓ Loaded mapping for {num_classes} classes")
    
    # Prepare Data
    dataloaders, dataset_sizes, class_names = create_dataloaders()
    
    # Build Model (MobileNetV2)
    print(f"\nBuilding MobileNetV2 Model...")
    
    if device.type == 'cuda':
        print(f"GPU Name: {torch.cuda.get_device_name(0)}")
        print(f"Memory Usage:")
        print(f"Allocated: {torch.cuda.memory_allocated(0)/1024**3:.1f} GB")
        print(f"Cached:    {torch.cuda.memory_reserved(0)/1024**3:.1f} GB")
        
    model_ft = models.mobilenet_v2(weights='DEFAULT')
    
    # Freeze parameters so we don't backprop through them
    for param in model_ft.parameters():
        param.requires_grad = False
        
    # Replace the classifier head
    # MobileNetV2 classifier is :
    # (classifier): Sequential(
    #    (0): Dropout(p=0.2, inplace=False)
    #    (1): Linear(in_features=1280, out_features=1000, bias=True)
    #  )
    num_ftrs = model_ft.classifier[1].in_features
    
    # New Head
    model_ft.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    model_ft = model_ft.to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Observe that only parameters of final layer are being optimized as
    # opposed to before.
    optimizer_ft = optim.Adam(model_ft.classifier.parameters(), lr=LEARNING_RATE)
    
    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    # Train Phase 1 (Heads only)
    print("\nStarting Training (Heads)...")
    model_ft, history = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           dataloaders, dataset_sizes, device, num_epochs=10)
    
    # Fine Tuning (Optional - unfreeze some layers)
    print("\nUnfreezing layers for Fine-Tuning...")
    for param in model_ft.parameters():
        param.requires_grad = True
        
    optimizer_ft = optim.Adam(model_ft.parameters(), lr=LEARNING_RATE/10)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    
    model_ft, history_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           dataloaders, dataset_sizes, device, num_epochs=EPOCHS-10)
                           
    # Combine histories if needed, or just plot the fine-tuning part
    
    # Save Final Model
    final_path = os.path.join(MODEL_SAVE_PATH, 'asl_model_final.pth')
    torch.save(model_ft.state_dict(), final_path)
    print(f"Final model saved to {final_path}")
    
    plot_history(history_ft, 'asl_model_finetuned')

if __name__ == '__main__':
    main()
