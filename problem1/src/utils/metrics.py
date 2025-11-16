"""Centralized metric computation utilities."""

import torch


class MetricsComputer:
    """Computes classification metrics (accuracy, top-k error, etc.)."""

    @staticmethod
    def compute_accuracy(outputs, targets, topk=(1,)):
        """
        Compute top-k accuracy.
        
        Args:
            outputs (torch.Tensor): Model output logits, shape (batch_size, num_classes)
            targets (torch.Tensor): Ground truth labels, shape (batch_size,)
            topk (tuple): Which top-k accuracies to compute (default: (1,))
        
        Returns:
            dict: Dictionary with 'top-k' keys and accuracy percentages as values
        """
        with torch.no_grad():
            batch_size = targets.size(0)
            results = {}
            
            for k in topk:
                _, pred = torch.topk(outputs, k, dim=1)
                pred = pred.t()
                correct = pred.eq(targets.view(1, -1).expand_as(pred))
                correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
                acc = correct_k.mul_(100.0 / batch_size).item()
                results[f"top-{k}"] = acc
            
            return results

    @staticmethod
    def compute_top_k_error(outputs, targets, topk=(1, 5,)):
        """
        Compute top-k error (percentage of samples where correct label is NOT in top-k).
        
        Args:
            outputs (torch.Tensor): Model output logits
            targets (torch.Tensor): Ground truth labels
            k (int): Top-k value (default: 5)
        
        Returns:
            float: Top-k error percentage
        """
        results = {}
        accuracy = MetricsComputer.compute_accuracy(outputs, targets, topk=topk)
        for k in topk:
            results[f"top-{k}-accuracy"] = accuracy[f"top-{k}"]
            results[f"top-{k}-error"] = 100.0 - accuracy[f"top-{k}"]
 
        return results

    @staticmethod
    def compute_confusion_matrix(outputs, targets, num_classes=100):
        """
        Compute confusion matrix.
        
        Args:
            outputs (torch.Tensor): Model output logits
            targets (torch.Tensor): Ground truth labels
            num_classes (int): Number of classes (default: 100)
        
        Returns:
            torch.Tensor: Confusion matrix of shape (num_classes, num_classes)
        """
        with torch.no_grad():
            _, preds = torch.max(outputs, 1)
            conf_matrix = torch.zeros(num_classes, num_classes)
            for t, p in zip(targets, preds):
                conf_matrix[t, p] += 1
            return conf_matrix
