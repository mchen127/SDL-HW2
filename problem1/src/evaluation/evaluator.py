"""Test set evaluation pipeline."""

import torch
from tqdm import tqdm
from src.utils.metrics import MetricsComputer


def evaluate(model, test_loader, device, return_predictions=False):
    """
    Run evaluation on the test set.
    
    Args:
        model (torch.nn.Module): Model to evaluate
        test_loader (torch.utils.data.DataLoader): Test DataLoader
        device (torch.device): Device to run evaluation on
        return_predictions (bool): If True, return predictions and targets (default: False)
    
    Returns:
        dict: Evaluation metrics:
            - 'top-1-accuracy': Top-1 accuracy
            - 'top-5-accuracy': Top-5 accuracy
            - 'top-5-error': Top-5 error
            - 'top-1-error': Top-1 error
            - 'confusion_matrix': Confusion matrix (if return_predictions=False)
        
        If return_predictions=True, returns:
            tuple: (metrics_dict, predictions, targets)
    """
    model.eval()
    
    all_outputs = []
    all_targets = []
    
    with torch.no_grad():
        progress_bar = tqdm(test_loader, desc="Evaluation", leave=False)
        for images, targets in progress_bar:
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            
            all_outputs.append(outputs.cpu())
            all_targets.append(targets.cpu())
    
    # Concatenate all outputs and targets
    all_outputs = torch.cat(all_outputs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics = MetricsComputer.compute_top_k_error(all_outputs, all_targets, topk=(1, 5))
    
    # Compute confusion matrix
    conf_matrix = MetricsComputer.compute_confusion_matrix(
        all_outputs, all_targets, num_classes=100
    )
    metrics["confusion_matrix"] = conf_matrix
    
    if return_predictions:
        return metrics, all_outputs, all_targets
    
    return metrics


class Evaluator:
    """
    Evaluator for running test-time evaluation on multiple subset sizes.
    """

    def __init__(self, device="cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize evaluator.
        
        Args:
            device (str or torch.device): Device to run evaluation on (default: 'cuda' if available)
        """
        if isinstance(device, str):
            self.device = torch.device(device)
        else:
            self.device = device

    def evaluate_model(self, model, test_loader, model_name="model", return_predictions=False):
        """
        Evaluate a single model on test set.
        
        Args:
            model (torch.nn.Module): Model to evaluate
            test_loader (torch.utils.data.DataLoader): Test DataLoader
            model_name (str): Name for logging (default: 'model')
            return_predictions (bool): If True, return predictions and targets
        
        Returns:
            dict: Evaluation metrics, optionally with predictions
        """
        print(f"\nEvaluating {model_name}...")
        
        if return_predictions:
            metrics, outputs, targets = evaluate(
                model,
                test_loader,
                self.device,
                return_predictions=True,
            )
            return metrics, outputs, targets
        else:
            metrics = evaluate(model, test_loader, self.device, return_predictions=False)
            
            # Print summary
            print(f"Results for {model_name}:")
            print(f"  Top-1 Accuracy: {metrics['top-1-accuracy']:.2f}%")
            print(f"  Top-5 Accuracy: {metrics['top-5-accuracy']:.2f}%")
            print(f"  Top-5 Error: {metrics['top-5-error']:.2f}%")
            print(f"  Top-1 Error: {metrics['top-1-error']:.2f}%")
            
            return metrics

    def evaluate_models_by_subset(self, models_dict, test_loader, verbose=True):
        """
        Evaluate multiple models (one per subset size).
        
        Args:
            models_dict (dict): Dictionary mapping subset_size -> model
            test_loader (torch.utils.data.DataLoader): Test DataLoader
            verbose (bool): If True, print results (default: True)
        
        Returns:
            dict: Dictionary mapping subset_size -> metrics
        """
        results = {}
        
        for subset_size, model in models_dict.items():
            metrics = self.evaluate_model(
                model,
                test_loader,
                model_name=f"Model (subset={subset_size})",
            )
            results[subset_size] = metrics
        
        return results
