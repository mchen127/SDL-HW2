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

    def __init__(self, checkpoint_dir, model_name="model", logs_dir=None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir (str or Path): Directory to save checkpoints
            model_name (str): Name prefix for checkpoint files
            logs_dir (str or Path, optional): Directory to save metrics/logs (defaults to checkpoint_dir)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir = Path(logs_dir) if logs_dir is not None else self.checkpoint_dir
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
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

        # Save periodic checkpoints (every 10 epochs)
        if epoch % 10 == 0:
            periodic_path = (
                self.checkpoint_dir / f"{self.model_name}_epoch_{epoch:03d}.pt"
            )
            torch.save(checkpoint, periodic_path)

    def load_checkpoint(
        self, model, optimizer=None, checkpoint_path=None, device="cpu"
    ):
        """
        Load a checkpoint.

        Args:
            model (torch.nn.Module): Model to load state into
            optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
                                                         Can be None for evaluation.
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

        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        if optimizer and "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint

    def _get_last_saved_epoch(self, filename, csv_path):
        """Determine last saved epoch for a CSV log (checks memory, then disk)."""
        if filename in self.last_saved_epoch:
            return self.last_saved_epoch[filename]
        if not csv_path.exists():
            return 0

        try:
            with open(csv_path, "r", newline="") as f:
                reader = csv.DictReader(f)
                last_row = None
                for row in reader:
                    last_row = row
                if last_row and "epoch" in last_row:
                    last_epoch = int(last_row["epoch"])
                    self.last_saved_epoch[filename] = last_epoch
                    return last_epoch
        except Exception as exc:
            print(f"Warning: Unable to read {csv_path} for last epoch: {exc}")

        return 0

    def append_metrics_rows(self, metrics_rows, filename="metrics.csv"):
        """Append multiple rows of metrics to a CSV file from a buffer."""
        if not metrics_rows:
            return

        csv_path = self.logs_dir / filename
        last_epoch = self._get_last_saved_epoch(filename, csv_path)

        # Filter out rows that might already be saved
        new_rows = [row for row in metrics_rows if row["epoch"] > last_epoch]
        if not new_rows:
            return

        # Determine fieldnames from the first new row
        fieldnames = list(new_rows[0].keys())
        if csv_path.exists():
            try:
                with open(csv_path, "r", newline="") as f:
                    reader = csv.DictReader(f)
                    if reader.fieldnames:
                        # Use existing header to maintain column order
                        fieldnames = reader.fieldnames
            except Exception as exc:
                print(f"Warning: Unable to read header from {csv_path}: {exc}")

        mode = "a" if csv_path.exists() else "w"
        with open(csv_path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if mode == "w":
                writer.writeheader()
            writer.writerows(new_rows)

        # Update the last saved epoch tracker
        self.last_saved_epoch[filename] = new_rows[-1]["epoch"]
