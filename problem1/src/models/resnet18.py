"""
Factory functions for creating ResNet-18 models using torchvision.
"""

import torch.nn as nn
from torchvision.models import resnet18


def create_resnet18(num_classes=100, pretrained=False):
    """
    Factory function to create ResNet-18 model from torchvision.

    This model can be pretrained on ImageNet.

    Args:
        num_classes (int): Number of output classes (default: 100 for mini-ImageNet).
        pretrained (bool): If True, load ImageNet pretrained weights (default: False).

    Returns:
        torch.nn.Module: ResNet-18 model from torchvision with a modified classifier.
    """
    if pretrained:
        model = resnet18(weights="IMAGENET1K_V1")
    else:
        model = resnet18(weights=None)

    # Modify the final fully connected layer for the specified number of classes
    if num_classes != 1000:
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model
