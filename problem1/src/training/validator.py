"""Validation logic and metric computation."""

import torch
import torch.nn as nn
from tqdm import tqdm
from src.utils.metrics import MetricsComputer


def validate(model, val_loader, device, criterion=None):
    """
    Run validation on the validation set.
    
    Args:
        model (torch.nn.Module): Model to validate
        val_loader (torch.utils.data.DataLoader): Validation DataLoader
        device (torch.device): Device to run validation on
        criterion (torch.nn.Module, optional): Loss function (default: None)
    
    Returns:
        dict: Dictionary with validation metrics:
            - 'loss': Average loss (if criterion provided)
            - 'top-1-accuracy': Top-1 accuracy
            - 'top-5-accuracy': Top-5 accuracy
            - 'top-1-error': Top-1 error (100 - top-1 accuracy)
            - 'top-5-error': Top-5 error (100 - top-5 accuracy)
    """
    model.eval()
    
    total_loss = 0.0
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc="Validation", leave=False)
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            # Compute loss if criterion provided
            if criterion is not None:
                loss = criterion(outputs, targets)
                total_loss += loss.item() * images.size(0)
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = MetricsComputer.compute_top_k_error(all_outputs, all_targets, topk=(1, 5))
    
    if criterion is not None:
        metrics["loss"] = total_loss / len(all_targets)
    
    return metrics
