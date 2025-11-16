"""Results handling and export utilities."""

import json
from pathlib import Path
import torch


class ResultsHandler:
    """
    Handles saving, loading, and organizing evaluation results.
    """

    def __init__(self, results_dir):
        """
        Initialize results handler.
        
        Args:
            results_dir (str or Path): Directory to save results
        """
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save_metrics(self, metrics, experiment_name, subset_size=None):
        """
        Save metrics to JSON file.
        
        Args:
            metrics (dict): Metrics dictionary
            experiment_name (str): Name of experiment
            subset_size (float or None): Subset size for organizing results
        """
        # Prepare data for JSON serialization (remove non-serializable items like confusion matrix)
        json_metrics = {}
        for key, value in metrics.items():
            if isinstance(value, torch.Tensor):
                # Skip tensors like confusion matrix
                continue
            else:
                json_metrics[key] = value
        
        # Create results file path
        if subset_size is not None:
            filename = f"metrics_{experiment_name}_subset_{subset_size}.json"
        else:
            filename = f"metrics_{experiment_name}.json"
        
        filepath = self.results_dir / filename
        
        with open(filepath, "w") as f:
            json.dump(json_metrics, f, indent=2)
        
        print(f"Saved metrics to {filepath}")
        return filepath

    def save_all_results(self, results_dict, experiment_name):
        """
        Save results for multiple subset sizes.
        
        Args:
            results_dict (dict): Dictionary mapping subset_size -> metrics
            experiment_name (str): Name of experiment
        """
        results_summary = {}
        
        for subset_size, metrics in results_dict.items():
            # Save individual results
            self.save_metrics(metrics, experiment_name, subset_size)
            
            # Add to summary
            summary_metrics = {}
            for key, value in metrics.items():
                if not isinstance(value, torch.Tensor):
                    summary_metrics[key] = value
            results_summary[str(subset_size)] = summary_metrics
        
        # Save summary
        summary_filepath = self.results_dir / f"results_{experiment_name}_summary.json"
        with open(summary_filepath, "w") as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"Saved results summary to {summary_filepath}")
        return summary_filepath

    def load_results(self, experiment_name):
        """
        Load results summary from JSON file.
        
        Args:
            experiment_name (str): Name of experiment
        
        Returns:
            dict: Results dictionary
        """
        summary_filepath = self.results_dir / f"results_{experiment_name}_summary.json"
        
        if not summary_filepath.exists():
            raise FileNotFoundError(f"Results file not found: {summary_filepath}")
        
        with open(summary_filepath, "r") as f:
            results = json.load(f)
        
        return results

    def export_for_plotting(self, results_dict, experiment_name):
        """
        Export results in a format suitable for plotting.
        
        Creates a JSON file with:
        - subset_sizes: List of subset sizes
        - top_5_errors: List of top-5 errors corresponding to subset sizes
        - top_1_accuracies: List of top-1 accuracies
        
        Args:
            results_dict (dict): Dictionary mapping subset_size -> metrics
            experiment_name (str): Name of experiment
        
        Returns:
            dict: Export dictionary
        """
        subset_sizes = []
        top_5_errors = []
        top_1_accuracies = []
        top_1_errors = []
        top_5_accuracies = []
        
        # Sort by subset size for proper ordering
        for subset_size in sorted(results_dict.keys()):
            metrics = results_dict[subset_size]
            subset_sizes.append(float(subset_size))
            top_5_errors.append(metrics.get("top-5-error", None))
            top_1_accuracies.append(metrics.get("top-1", None))
            top_1_errors.append(metrics.get("top-1-error", None))
            top_5_accuracies.append(metrics.get("top-5", None))
        
        export_data = {
            "subset_sizes": subset_sizes,
            "top_5_errors": top_5_errors,
            "top_1_accuracies": top_1_accuracies,
            "top_1_errors": top_1_errors,
            "top_5_accuracies": top_5_accuracies,
        }
        
        # Save to file
        export_filepath = self.results_dir / f"results_{experiment_name}_export.json"
        with open(export_filepath, "w") as f:
            json.dump(export_data, f, indent=2)
        
        print(f"Exported plotting data to {export_filepath}")
        return export_data

    @staticmethod
    def load_plotting_data(filepath):
        """
        Load plotting data from JSON file.
        
        Args:
            filepath (str or Path): Path to results file
        
        Returns:
            dict: Plotting data with subset_sizes and metrics
        """
        with open(filepath, "r") as f:
            return json.load(f)
