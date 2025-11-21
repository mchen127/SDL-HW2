"""Model architectures for image classification."""

from .alexnet import create_alexnet
from .resnet18 import create_resnet18

__all__ = ["create_alexnet", "create_resnet18"]
