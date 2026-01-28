from torchvision import transforms

def get_transform():
    """
    Returns the image transformation pipeline.
    Matches the validation transforms used during training:
    - Resize to 224x224
    - Convert to Tensor
    - Normalize with ImageNet mean/std
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
