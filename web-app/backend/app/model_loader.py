import torch
import torch.nn as nn
from torchvision import models

def load_model_architecture(num_classes):
    """
    Recreates the exact MobileNetV2 architecture used in training.
    Must match lines 208-225 of your 3_train_model.py
    """
    # Load base model structure
    # weights=None because we will load our state_dict later
    model_ft = models.mobilenet_v2(weights=None) 
    
    # Get input features of the classifier
    num_ftrs = model_ft.classifier[1].in_features
    
    # Recreate the custom head exactly as defined in training
    model_ft.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(256, num_classes)
    )
    
    return model_ft
