"""Data augmentation and preprocessing transforms for mini-ImageNet."""

from torchvision import transforms
from PIL import Image


class RGBConverter:
    """Convert grayscale images to RGB."""
    
    def __call__(self, img):
        """Convert image to RGB if needed."""
        if isinstance(img, Image.Image):
            if img.mode != 'RGB':
                img = img.convert('RGB')
        return img


def get_train_transform(input_size=227):
    """
    Get training data transforms with augmentation.
    
    Args:
        input_size (int): Target image size (default: 227 for AlexNet)
    
    Returns:
        torchvision.transforms.Compose: Training transform pipeline
    """
    return transforms.Compose([
        RGBConverter(),  # Convert grayscale to RGB
        transforms.Resize((input_size, input_size)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])


def get_eval_transform(input_size=227):
    """
    Get evaluation (validation/test) data transforms without augmentation.
    
    Args:
        input_size (int): Target image size (default: 227 for AlexNet)
    
    Returns:
        torchvision.transforms.Compose: Evaluation transform pipeline
    """
    return transforms.Compose([
        RGBConverter(),  # Convert grayscale to RGB
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225],   # ImageNet std
        ),
    ])
