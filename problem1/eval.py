"""Evaluation script for test set evaluation and results collection."""

import argparse
import torch
from pathlib import Path
import json

from src.config import get_config
from src.models import create_alexnet, create_alexnet_torchvision
from src.data import get_test_loader
from src.evaluation import Evaluator
from src.evaluation.results_handler import ResultsHandler
from src.training.checkpoint import CheckpointManager


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate AlexNet on mini-ImageNet test set"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        choices=["alexnet", "alexnet_torchvision"],
        default="alexnet",
        help="Model architecture to evaluate (default: alexnet)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Directory containing training checkpoints",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./results",
        help="Directory to save evaluation results (default: ./results)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for evaluation (default: 128)",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cuda", "cpu", "cuda:0", "cuda:1"],
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (default: cuda if available)",
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


def evaluate_subset(
    model_name,
    subset_size,
    checkpoint_dir,
    test_loader,
    device,
    results_handler,
):
    """
    Evaluate model trained on a specific subset size.
    
    Args:
        model_name (str): Model architecture name
        subset_size (float): Subset size used for training
        checkpoint_dir (Path): Directory containing checkpoints
        test_loader (torch.utils.data.DataLoader): Test DataLoader
        device (torch.device): Device to run evaluation on
        results_handler (ResultsHandler): Results handler for saving
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"\nEvaluating subset size: {subset_size}")
    
    # Create model
    model = create_model(model_name, num_classes=100)
    
    # Load checkpoint
    subset_checkpoint_dir = checkpoint_dir / f"subset_{subset_size:.4f}"
    checkpoint_manager = CheckpointManager(subset_checkpoint_dir, model_name)
    
    best_checkpoint_path = subset_checkpoint_dir / f"{model_name}_best.pt"
    if not best_checkpoint_path.exists():
        print(f"Warning: Best checkpoint not found at {best_checkpoint_path}")
        print(f"Attempting to find any checkpoint in {subset_checkpoint_dir}")
        # List available checkpoints
        pt_files = list(subset_checkpoint_dir.glob("*.pt"))
        if not pt_files:
            raise FileNotFoundError(f"No checkpoints found in {subset_checkpoint_dir}")
        best_checkpoint_path = pt_files[0]
        print(f"Using checkpoint: {best_checkpoint_path}")
    
    checkpoint = torch.load(best_checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Evaluate
    evaluator = Evaluator(device=device)
    metrics = evaluator.evaluate_model(
        model,
        test_loader,
        model_name=f"{model_name} (subset={subset_size})",
    )
    
    # Save metrics
    results_handler.save_metrics(metrics, model_name, subset_size)
    
    return metrics


def main():
    """Main evaluation function."""
    args = parse_args()
    config = get_config()
    
    # Set device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(config["device"]["device"])
    print(f"Using device: {device}")
    
    # Set checkpoint and results directories
    checkpoint_dir = Path(args.checkpoint_dir)
    if not checkpoint_dir.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    print(f"Checkpoint directory: {checkpoint_dir}")
    
    if args.results_dir:
        results_dir = Path(args.results_dir)
    else:
        results_dir = config["paths"]["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results directory: {results_dir}")
    
    # Create test loader
    test_loader = get_test_loader(
        batch_size=args.batch_size,
        num_workers=config["data"]["num_workers"],
        pin_memory=config["data"]["pin_memory"],
        input_size=config["model"]["input_size"],
    )
    print(f"Test loader: {len(test_loader)} batches")
    
    # Create results handler
    results_handler = ResultsHandler(results_dir)
    
    # Find all subset directories
    subset_dirs = sorted(checkpoint_dir.glob("subset_*"))
    if not subset_dirs:
        raise FileNotFoundError(f"No subset directories found in {checkpoint_dir}")
    
    # Extract subset sizes
    subset_sizes = []
    for subset_dir in subset_dirs:
        # Extract subset size from dirname (e.g., "subset_0.5000")
        subset_str = subset_dir.name.replace("subset_", "")
        try:
            subset_size = float(subset_str)
            subset_sizes.append(subset_size)
        except ValueError:
            print(f"Warning: Could not parse subset size from {subset_dir.name}")
    
    subset_sizes = sorted(subset_sizes)
    print(f"Subset sizes found: {subset_sizes}")
    
    # Evaluate on each subset
    all_results = {}
    for subset_size in subset_sizes:
        try:
            metrics = evaluate_subset(
                model_name=args.model,
                subset_size=subset_size,
                checkpoint_dir=checkpoint_dir,
                test_loader=test_loader,
                device=device,
                results_handler=results_handler,
            )
            all_results[subset_size] = metrics
        except Exception as e:
            print(f"Error evaluating subset {subset_size}: {e}")
            import traceback
            traceback.print_exc()
    
    # Save results summary and export for plotting
    if all_results:
        results_handler.save_all_results(all_results, args.model)
        plotting_data = results_handler.export_for_plotting(all_results, args.model)
        
        print(f"\n{'='*80}")
        print("Evaluation completed")
        print(f"Results saved to: {results_dir}")
        print(f"{'='*80}")
        
        # Print summary
        print("\nResults Summary:")
        print(f"{'Subset Size':<15} {'Top-1 Acc':<15} {'Top-5 Error':<15}")
        print("-" * 45)
        for subset_size in sorted(all_results.keys()):
            metrics = all_results[subset_size]
            top1 = metrics.get("top-1", 0)
            top5_err = metrics.get("top-5-error", 0)
            print(f"{subset_size:<15.4f} {top1:<15.2f} {top5_err:<15.2f}")
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
