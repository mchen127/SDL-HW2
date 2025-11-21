"""
Factory functions for creating AlexNet models using torchvision.
"""

import torch.nn as nn
from torchvision.models import alexnet

def create_alexnet(num_classes=100, pretrained=False):
    """
    Factory function to create AlexNet model from torchvision.

    This model can be pretrained on ImageNet, but for fair comparison with other
    models, we often train from scratch on the target dataset.

    Args:
        num_classes (int): Number of output classes (default: 100 for mini-ImageNet).
        pretrained (bool): If True, load ImageNet pretrained weights (default: False).

    Returns:
        torch.nn.Module: AlexNet model from torchvision with a modified classifier.
    """
    if pretrained:
        model = alexnet(weights="IMAGENET1K_V1")
    else:
        model = alexnet(weights=None)

    # Modify the classifier for the specified number of classes
    if num_classes != 1000:
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)

    return model