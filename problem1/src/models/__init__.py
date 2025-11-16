"""Model architectures for image classification."""

from .alexnet import AlexNet, create_alexnet, create_alexnet_torchvision

__all__ = ["AlexNet", "create_alexnet", "create_alexnet_torchvision"]
