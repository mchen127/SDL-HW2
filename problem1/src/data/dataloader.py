"""DataLoader utilities for mini-ImageNet."""

from torch.utils.data import DataLoader
from .dataset import MiniImageNetDataset
from .transforms import get_train_transform, get_eval_transform


def get_train_loader(
    subset_size=1.0,
    batch_size=128,
    num_workers=16,
    pin_memory=True,
    random_state=42,
    input_size=227,
):
    """
    Create training DataLoader with augmentation.
    
    Args:
        subset_size (float or int): Fraction or count of training data to use
        batch_size (int): Batch size (default: 128)
        num_workers (int): Number of data loading workers (default: 4)
        pin_memory (bool): Pin memory for faster GPU transfer (default: True)
        random_state (int): Random seed (default: 42)
        input_size (int): Target image size (default: 227)
    
    Returns:
        torch.utils.data.DataLoader: Training DataLoader with shuffling and augmentation
    """
    dataset = MiniImageNetDataset(
        split="train",
        subset_size=subset_size,
        transform=get_train_transform(input_size),
        random_state=random_state,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )


def get_val_loader(
    batch_size=128,
    num_workers=16,
    pin_memory=True,
    input_size=227,
):
    """
    Create validation DataLoader without augmentation.
    
    Args:
        batch_size (int): Batch size (default: 128)
        num_workers (int): Number of data loading workers (default: 4)
        pin_memory (bool): Pin memory for faster GPU transfer (default: True)
        input_size (int): Target image size (default: 227)
    
    Returns:
        torch.utils.data.DataLoader: Validation DataLoader without shuffling or augmentation
    """
    dataset = MiniImageNetDataset(
        split="validation",
        subset_size=1.0,
        transform=get_eval_transform(input_size),
        random_state=42,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )


def get_test_loader(
    batch_size=128,
    num_workers=16,
    pin_memory=True,
    input_size=227,
):
    """
    Create test DataLoader without augmentation.
    
    Args:
        batch_size (int): Batch size (default: 128)
        num_workers (int): Number of data loading workers (default: 4)
        pin_memory (bool): Pin memory for faster GPU transfer (default: True)
        input_size (int): Target image size (default: 227)
    
    Returns:
        torch.utils.data.DataLoader: Test DataLoader without shuffling or augmentation
    """
    dataset = MiniImageNetDataset(
        split="test",
        subset_size=1.0,
        transform=get_eval_transform(input_size),
        random_state=42,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
