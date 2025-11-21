"""Default configuration for AlexNet training on mini-ImageNet."""

import torch
from pathlib import Path

# Get project root
PROJECT_ROOT = Path(__file__).parent.parent.parent

# ============================================================================
# Model Configuration
# ============================================================================
MODEL_CONFIG = {
    "name": "alexnet",  # 'alexnet' or 'alexnet_torchvision'
    "num_classes": 100,  # mini-ImageNet has 100 classes
    "dropout_p": 0.5,  # Dropout probability in FC layers
    "input_size": 227,  # AlexNet input size
}

# ============================================================================
# Data Configuration
# ============================================================================
DATA_CONFIG = {
    "dataset_name": "timm/mini-imagenet",
    "batch_size": 128,
    "num_workers": 64,
    "pin_memory": True,
    "subset_sizes": [1.0, 0.5, 0.25, 0.125, 0.0625],  # Geometric progression
    "random_state": 42,
}

# ============================================================================
# Training Configuration
# ============================================================================
TRAINING_CONFIG = {
    "num_epochs": 100,
    "learning_rate": 0.01,
    "momentum": 0.9,
    "weight_decay": 5e-4,
    "patience": 5,  # Early stopping patience (None = no early stopping)
    "log_interval": 10,  # How often to flush metric logs to disk (in epochs)
}

# ============================================================================
# Device Configuration
# ============================================================================
DEVICE_CONFIG = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
}

# ============================================================================
# Paths Configuration
# ============================================================================
PATHS_CONFIG = {
    "project_root": PROJECT_ROOT,
    "checkpoints_dir": PROJECT_ROOT / "checkpoints",
    "logs_dir": PROJECT_ROOT / "logs",
}

# Ensure directories exist
for dir_path in [PATHS_CONFIG["checkpoints_dir"], PATHS_CONFIG["logs_dir"]]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Consolidated Config
# ============================================================================
CONFIG = {
    "model": MODEL_CONFIG,
    "data": DATA_CONFIG,
    "training": TRAINING_CONFIG,
    "device": DEVICE_CONFIG,
    "paths": PATHS_CONFIG,
}


def get_config():
    """
    Get the default configuration dictionary.
    
    Returns:
        dict: Configuration dictionary
    """
    return CONFIG


def update_config(updates):
    """
    Update configuration with custom values.
    
    Args:
        updates (dict): Dictionary of configuration updates.
                       Should follow the structure of CONFIG.
    
    Example:
        >>> config = get_config()
        >>> update_config({
        ...     "training": {"learning_rate": 0.001}
        ... })
    """
    def recursive_update(config_dict, updates_dict):
        for key, value in updates_dict.items():
            if isinstance(value, dict) and key in config_dict:
                recursive_update(config_dict[key], value)
            else:
                config_dict[key] = value
    
    recursive_update(CONFIG, updates)
