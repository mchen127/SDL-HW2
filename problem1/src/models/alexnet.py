"""AlexNet implementation adapted for mini-ImageNet."""

import torch
import torch.nn as nn
from torchvision.models import alexnet as torchvision_alexnet


class AlexNet(nn.Module):
    """
    AlexNet architecture adapted for mini-ImageNet classification.
    
    Original AlexNet from Krizhevsky et al. (2012) with modifications:
    - Input: 227×227 (original specification)
    - Output: 100 classes (mini-ImageNet)
    - Dropout: p=0.5 in FC layers for regularization
    
    Architecture:
    - 5 convolutional layers with ReLU activations and max pooling
    - 3 fully connected layers with ReLU
    - Final output layer: softmax for 100-class classification
    
    Reference:
        Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012).
        ImageNet classification with deep convolutional neural networks.
        Advances in neural information processing systems, 25.
    """

    def __init__(self, num_classes=100, dropout_p=0.5):
        """
        Initialize AlexNet.
        
        Args:
            num_classes (int): Number of output classes (default: 100 for mini-ImageNet)
            dropout_p (float): Dropout probability in FC layers (default: 0.5)
        """
        super(AlexNet, self).__init__()
        
        self.num_classes = num_classes
        self.dropout_p = dropout_p
        
        # Convolutional layers
        self.features = nn.Sequential(
            # Layer 1: Conv + ReLU + MaxPool
            # Input: 227×227×3
            # Output: 55×55×64
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 2: Conv + ReLU + MaxPool
            # Output: 27×27×192
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Layer 3: Conv + ReLU
            # Output: 27×27×384
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 4: Conv + ReLU
            # Output: 27×27×256
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Layer 5: Conv + ReLU + MaxPool
            # Output: 13×13×256
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        # Global average pooling to handle variable input sizes
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        # Fully connected layers
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout_p),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Kaiming normal distribution for conv and linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Forward pass through AlexNet.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, 3, 227, 227)
        
        Returns:
            torch.Tensor: Output logits of shape (batch_size, num_classes)
        """
        # Convolutional feature extraction
        x = self.features(x)
        
        # Global average pooling
        x = self.avgpool(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        
        return x


def create_alexnet(num_classes=100, pretrained=False):
    """
    Factory function to create AlexNet model.
    
    Args:
        num_classes (int): Number of output classes (default: 100)
        pretrained (bool): If True, load pretrained weights (not implemented; 
                          train from scratch for this task)
    
    Returns:
        AlexNet: Initialized model
    """
    if pretrained:
        raise NotImplementedError(
            "Pretrained weights not available for this implementation. "
            "Train from scratch for reproducibility."
        )
    
    return AlexNet(num_classes=num_classes)


def create_alexnet_torchvision(num_classes=100, pretrained=False):
    """
    Factory function to create AlexNet model from torchvision.
    
    Uses the torchvision implementation of AlexNet for comparison with custom implementation.
    This model is pretrained on ImageNet by default, but for fair comparison with custom
    implementation, we train from scratch on mini-ImageNet.
    
    Args:
        num_classes (int): Number of output classes (default: 100 for mini-ImageNet)
        pretrained (bool): If True, load ImageNet pretrained weights (default: False).
                          For learning curve analysis, typically set to False for fair comparison.
    
    Returns:
        torch.nn.Module: AlexNet model from torchvision with modified classifier
    """
    # Load torchvision AlexNet (pretrained on ImageNet or random init)
    if pretrained:
        model = torchvision_alexnet(weights="IMAGENET1K_V1")
    else:
        model = torchvision_alexnet(weights=None)
    
    # Modify classifier for 100 classes
    # Original torchvision AlexNet has classifier output size of 1000
    if num_classes != 1000:
        # Replace the final FC layer to output num_classes
        model.classifier[-1] = nn.Linear(4096, num_classes)
    
    return model
