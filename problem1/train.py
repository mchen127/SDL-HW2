"""Main training script for AlexNet on mini-ImageNet."""

import argparse
import torch
from pathlib import Path
from datetime import datetime

from src.config import get_config, update_config
from src.models import create_alexnet, create_alexnet_torchvision
from src.data import get_train_loader, get_val_loader
from src.training import Trainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train AlexNet on mini-ImageNet with varying dataset sizes"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["alexnet", "alexnet_torchvision"],
        default="alexnet",
        help="Model architecture to use (default: alexnet)",
    )
    parser.add_argument(
        "--subset-sizes",
        type=float,
        nargs="+",
        default=[1.0, 0.5, 0.25, 0.125, 0.0625],
        help="Subset sizes to train on (default: [1.0, 0.5, 0.25, 0.125, 0.0625])",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size (default: 128)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Learning rate (default: 0.01)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 90)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=5,
        help="Early stopping patience (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "cuda:0", "cuda:1"],
        default=None,
        help="Device to use (default: cuda if available)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory to save checkpoints (default: ./checkpoints)",
    )
    
    return parser.parse_args()


def create_model(model_name, num_classes=100):
    """
    Create model instance.
    
    Args:
        model_name (str): Model name ('alexnet' or 'alexnet_torchvision')
        num_classes (int): Number of output classes
    
    Returns:
        torch.nn.Module: Model instance
    """
    if model_name == "alexnet":
        return create_alexnet(num_classes=num_classes)
    elif model_name == "alexnet_torchvision":
        return create_alexnet_torchvision(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def train_single_subset(
    model_name,
    subset_size,
    config,
    device,
    checkpoint_dir,
):
    """
    Train a single model on a specific subset size.
    
    Args:
        model_name (str): Model architecture name
        subset_size (float): Fraction of training data to use
        config (dict): Configuration dictionary
        device (torch.device): Device to train on
        checkpoint_dir (Path): Directory to save checkpoints
    
    Returns:
        dict: Training history
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name} on subset size: {subset_size}")
    print(f"{'='*80}")
    
    # Create data loaders
    train_loader = get_train_loader(
        subset_size=subset_size,
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        random_state=config["data"]["random_state"],
        input_size=config["model"]["input_size"],
    )
    
    val_loader = get_val_loader(
        batch_size=config["data"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        input_size=config["model"]["input_size"],
    )
    
    print(f"Train loader: {len(train_loader)} batches")
    print(f"Val loader: {len(val_loader)} batches")
    
    # Create model
    model = create_model(model_name, num_classes=config["model"]["num_classes"])
    print(f"Model created: {model_name}")
    
    # Create trainer
    subset_checkpoint_dir = checkpoint_dir / f"subset_{subset_size:.4f}"
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        checkpoint_dir=subset_checkpoint_dir,
        model_name=model_name,
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Train
    history = trainer.train(
        num_epochs=config["training"]["num_epochs"],
        patience=config["training"]["patience"],
    )
    
    print(f"Training completed for subset size: {subset_size}")
    
    return history


def main():
    """Main training function."""
    args = parse_args()
    config = get_config()
    
    # Update config with command line arguments
    updates = {}
    if args.batch_size is not None:
        updates.setdefault("data", {})["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        updates.setdefault("training", {})["learning_rate"] = args.learning_rate
    if args.num_epochs is not None:
        updates.setdefault("training", {})["num_epochs"] = args.num_epochs
    if args.patience is not None:
        updates.setdefault("training", {})["patience"] = args.patience
    if args.device is not None:
        updates.setdefault("device", {})["device"] = args.device
    
    if updates:
        update_config(updates)
        config = get_config()
    
    # Set device
    device = torch.device(config["device"]["device"])
    print(f"Using device: {device}")
    
    # Determine subset sizes
    subset_sizes = args.subset_sizes if args.subset_sizes else config["data"]["subset_sizes"]
    print(f"Subset sizes: {subset_sizes}")
    
    # Determine checkpoint directory
    if args.checkpoint_dir:
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = config["paths"]["checkpoints_dir"] / args.model / timestamp
    
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    # Train on each subset size
    all_histories = {}
    for subset_size in sorted(subset_sizes, reverse=True):  # Start with largest subset
        try:
            history = train_single_subset(
                model_name=args.model,
                subset_size=subset_size,
                config=config,
                device=device,
                checkpoint_dir=checkpoint_dir,
            )
            all_histories[subset_size] = history
        except Exception as e:
            print(f"Error training on subset {subset_size}: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n{'='*80}")
    print("Training completed for all subset sizes")
    print(f"Results saved to: {checkpoint_dir}")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()
