"""Checkpoint management for model training."""

import os
import torch
import json
import csv
from pathlib import Path


class CheckpointManager:
    """
    Manages model checkpoints during training.
    
    Saves model state, optimizer state, and metrics at intervals,
    with support for best model tracking and resumable training.
    """

    def __init__(self, checkpoint_dir, model_name="model", keep_best_only=True):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir (str or Path): Directory to save checkpoints
            model_name (str): Name prefix for checkpoint files
            keep_best_only (bool): If True, keep only the best checkpoint by validation metric
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.keep_best_only = keep_best_only
        self.best_metric = None
        self.best_checkpoint_path = None
        self.last_saved_epoch = {}  # Track last saved epoch per CSV file

    def save_checkpoint(
        self,
        epoch,
        model,
        optimizer,
        metrics,
        is_best=False,
    ):
        """
        Save a checkpoint.
        
        Args:
            epoch (int): Current epoch number
            model (torch.nn.Module): Model to save
            optimizer (torch.optim.Optimizer): Optimizer to save
            metrics (dict): Metrics dictionary to save (e.g., {'loss': 0.5, 'top1_acc': 0.9})
            is_best (bool): Whether this is the best checkpoint so far
        """
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }
        
        # Save latest checkpoint
        latest_path = self.checkpoint_dir / f"{self.model_name}_latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save best checkpoint if applicable
        if is_best:
            best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
            torch.save(checkpoint, best_path)
            self.best_checkpoint_path = best_path
            
            # Remove previous best if keeping only best
            if self.keep_best_only and best_path != latest_path:
                # Best is already saved; latest will be overwritten next epoch
                pass
        
        # Save periodic checkpoints (every 10 epochs)
        if epoch % 10 == 0:
            periodic_path = self.checkpoint_dir / f"{self.model_name}_epoch_{epoch:03d}.pt"
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(self, model, optimizer, checkpoint_path=None):
        """
        Load a checkpoint.
        
        Args:
            model (torch.nn.Module): Model to load state into
            optimizer (torch.optim.Optimizer): Optimizer to load state into
            checkpoint_path (str or Path, optional): Path to checkpoint. 
                                                     If None, loads best checkpoint.
        
        Returns:
            dict: Loaded checkpoint containing epoch and metrics
        
        Raises:
            FileNotFoundError: If checkpoint file not found
        """
        if checkpoint_path is None:
            if self.best_checkpoint_path is None:
                best_path = self.checkpoint_dir / f"{self.model_name}_best.pt"
                if not best_path.exists():
                    raise FileNotFoundError(f"No best checkpoint found at {best_path}")
                checkpoint_path = best_path
            else:
                checkpoint_path = self.best_checkpoint_path
        
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        return checkpoint

    def save_metrics(self, metrics_dict, filename="metrics.json"):
        """
        Save metrics history to JSON file.
        
        Args:
            metrics_dict (dict): Dictionary of metrics to save
            filename (str): Name of metrics file (default: "metrics.json")
        """
        metrics_path = self.checkpoint_dir / filename
        with open(metrics_path, "w") as f:
            json.dump(metrics_dict, f, indent=2)

    def load_metrics(self, filename="metrics.json"):
        """
        Load metrics history from JSON file.
        
        Args:
            filename (str): Name of metrics file (default: "metrics.json")
        
        Returns:
            dict: Loaded metrics dictionary
        """
        metrics_path = self.checkpoint_dir / filename
        if not metrics_path.exists():
            return {}
        
        with open(metrics_path, "r") as f:
            return json.load(f)

    def save_metrics_csv(self, metrics_dict, filename="metrics.csv"):
        """
        Save metrics history to CSV file (flattened format).
        Only appends new rows since the last save.
        
        Args:
            metrics_dict (dict): Dictionary with 'train' and 'val' keys containing lists of metrics
            filename (str): Name of CSV file (default: "metrics.csv")
        """
        csv_path = self.checkpoint_dir / filename
        
        # Flatten metrics: combine train and val metrics per epoch
        rows = []
        train_metrics = metrics_dict.get("train", [])
        val_metrics = metrics_dict.get("val", [])
        
        for epoch, (train_m, val_m) in enumerate(zip(train_metrics, val_metrics), start=1):
            row = {"epoch": epoch}
            # Add train metrics with prefix
            for key, value in train_m.items():
                row[f"train_{key}"] = value
            # Add val metrics with prefix
            for key, value in val_m.items():
                row[f"val_{key}"] = value
            rows.append(row)
        
        if not rows:
            print(f"Warning: No metrics to save to CSV")
            return
        
        # Get last saved epoch for this file (default 0 if first time)
        last_epoch = self.last_saved_epoch.get(filename, 0)
        new_rows = rows[last_epoch:]  # Only get rows after the last saved epoch
        
        if not new_rows:
            return  # Nothing new to append
        
        fieldnames = list(rows[0].keys())
        
        if csv_path.exists():
            # Append mode: only write new rows
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerows(new_rows)
        else:
            # First time: write header + all rows up to current
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        
        # Update last saved epoch
        self.last_saved_epoch[filename] = len(rows)
