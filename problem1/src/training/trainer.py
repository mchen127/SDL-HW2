"""Main training loop and trainer class."""

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from pathlib import Path
import json

from .checkpoint import CheckpointManager
from .validator import validate
from src.utils.metrics import MetricsComputer


class Trainer:
    """
    Trainer class for training neural networks on mini-ImageNet.

    Handles:
    - Training loop with per-epoch validation
    - Loss and metric tracking
    - Checkpoint saving (best model + periodic)
    - Early stopping based on validation metrics
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device,
        checkpoint_dir,
        model_name="model",
        lr=0.01,
        momentum=0.9,
        weight_decay=5e-4,
        logs_dir=None,
        log_interval=10,
    ):
        """
        Initialize trainer.

        Args:
            model (torch.nn.Module): Model to train
            train_loader (torch.utils.data.DataLoader): Training DataLoader
            val_loader (torch.utils.data.DataLoader): Validation DataLoader
            device (torch.device): Device to train on
            checkpoint_dir (str or Path): Directory to save checkpoints
            model_name (str): Name for checkpoint files
            lr (float): Learning rate (default: 0.01)
            momentum (float): SGD momentum (default: 0.9)
            weight_decay (float): L2 regularization (default: 5e-4)
            logs_dir (str or Path, optional): Directory to save training logs/metrics
            log_interval (int): How often to flush metric logs to disk (in epochs)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.logs_dir = Path(logs_dir) if logs_dir is not None else Path(checkpoint_dir)

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = optim.SGD(
            self.model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )

        # Checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=checkpoint_dir,
            model_name=model_name,
            logs_dir=self.logs_dir,
        )

        # Metrics tracking
        self.train_history = []
        self.val_history = []
        self.metrics_buffer = []
        self.log_interval = log_interval

    def train_epoch(self):
        """
        Run one epoch of training.

        Returns:
            dict: Training metrics for this epoch (loss, top-1 acc, top-5 acc)
        """
        self.model.train()

        total_loss = 0.0
        all_outputs = []
        all_targets = []

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, targets in progress_bar:
            images = images.to(self.device)
            targets = targets.to(self.device)

            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, targets)

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Accumulate metrics
            total_loss += loss.item() * images.size(0)
            all_outputs.append(outputs.detach().cpu())
            all_targets.append(targets.cpu())

            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute epoch metrics
        all_outputs = torch.cat(all_outputs, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        metrics = MetricsComputer.compute_top_k_error(
            all_outputs, all_targets, topk=(1, 5)
        )
        metrics["loss"] = total_loss / len(all_targets)

        return metrics

    def _flush_metrics_buffer(self):
        """Write buffered metrics to CSV and clear the buffer."""
        if not self.metrics_buffer:
            return

        print(f"Flushing {len(self.metrics_buffer)} log entries to disk...")
        self.checkpoint_manager.append_metrics_rows(
            self.metrics_buffer, filename="metrics.csv"
        )
        self.metrics_buffer.clear()

    def load_best_model(self):
        """Load the best checkpoint into the model."""
        checkpoint = self.checkpoint_manager.load_checkpoint(
            self.model,
            self.optimizer,
        )
        print(f"Loaded best model from epoch {checkpoint['epoch']}")

    def train(self, num_epochs, patience=None):
        """
        Train the model for multiple epochs.

        Args:
            num_epochs (int): Number of epochs to train
            patience (int, optional): Early stopping patience (number of epochs with no improvement).
                                     If None, no early stopping is applied.

        Returns:
            dict: Training and validation history
        """
        best_val_loss = float("inf")
        patience_counter = 0

        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")

            # Training and validation
            train_metrics = self.train_epoch()
            val_metrics = validate(
                self.model, self.val_loader, self.device, self.criterion
            )

            # Print metrics and update history
            self._log_epoch_metrics(epoch, train_metrics, val_metrics)

            # Checkpoint management
            is_best = val_metrics["loss"] < best_val_loss
            if is_best:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
            else:
                patience_counter += 1

            self.checkpoint_manager.save_checkpoint(
                epoch=epoch,
                model=self.model,
                optimizer=self.optimizer,
                metrics={"train": train_metrics, "val": val_metrics},
                is_best=is_best,
            )

            # Early stopping
            if patience is not None and patience_counter >= patience:
                print(f"\nEarly stopping triggered (patience={patience})")
                break

        # Final flush of any remaining logs and return history
        self._flush_metrics_buffer()
        return {"train": self.train_history, "val": self.val_history}

    def _log_epoch_metrics(self, epoch, train_metrics, val_metrics):
        """Logs metrics to console, history, and buffer."""
        self.train_history.append(train_metrics)
        self.val_history.append(val_metrics)

        print(
            f"Train Loss: {train_metrics['loss']:.4f} | Train Top-1-Error: {train_metrics['top-1-error']:.2f}% | Train Top-5-Error: {train_metrics['top-5-error']:.2f}%"
        )
        print(
            f"Val Loss: {val_metrics['loss']:.4f} | Val Top-1-Error: {val_metrics['top-1-error']:.2f}% | Val Top-5-Error: {val_metrics['top-5-error']:.2f}%"
        )

        # Add metrics to buffer
        row = {"epoch": epoch}
        row.update({f"train_{k}": v for k, v in train_metrics.items()})
        row.update({f"val_{k}": v for k, v in val_metrics.items()})
        self.metrics_buffer.append(row)

        # Flush buffer periodically
        if epoch % self.log_interval == 0:
            self._flush_metrics_buffer()
