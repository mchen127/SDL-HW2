"""Mini-ImageNet dataset implementation with subset sampling."""

import torch
from torch.utils.data import Dataset
from datasets import load_dataset
import numpy as np
from collections import defaultdict


class MiniImageNetDataset(Dataset):
    """
    Mini-ImageNet dataset wrapper with support for stratified subset sampling.
    
    Attributes:
        split (str): Dataset split ('train', 'validation', or 'test')
        subset_size (float or int): Fraction (0-1) or absolute count of samples to use
        transform: Data augmentation/preprocessing pipeline
        random_state (int): Seed for reproducibility
    """

    def __init__(
        self,
        split="train",
        subset_size=1.0,
        transform=None,
        random_state=42,
    ):
        """
        Initialize Mini-ImageNet dataset.
        
        Args:
            split (str): Dataset split ('train', 'validation', or 'test')
            subset_size (float or int): 
                - If float in (0, 1]: fraction of dataset to use (e.g., 0.5 for 50%)
                - If int > 1: absolute number of samples
                Default: 1.0 (full dataset)
            transform: Callable to apply to images (e.g., torchvision.transforms)
            random_state (int): Random seed for reproducibility
        """
        assert split in ["train", "validation", "test"], f"Invalid split: {split}"
        assert subset_size > 0, f"subset_size must be positive, got {subset_size}"
        
        self.split = split
        self.transform = transform
        self.random_state = random_state
        
        # Load full dataset
        self.full_dataset = load_dataset("timm/mini-imagenet")[split]
        
        # Get stratified subset
        self.indices = self._get_stratified_subset(subset_size)
        
    def _get_stratified_subset(self, subset_size):
        """
        Get stratified subset indices preserving class distribution.
        
        Args:
            subset_size (float or int): Fraction or absolute count
        
        Returns:
            list: Indices of samples in the subset
        """
        np.random.seed(self.random_state)
        
        # Get all labels
        labels = np.array(self.full_dataset["label"])
        num_samples = len(labels)
        
        # Determine target subset size
        if 0 < subset_size <= 1:
            target_size = int(num_samples * subset_size)
        else:
            target_size = int(subset_size)
        
        target_size = min(target_size, num_samples)
        
        # Stratified sampling: sample equally from each class
        indices_by_class = defaultdict(list)
        for idx, label in enumerate(labels):
            indices_by_class[label].append(idx)
        
        subset_indices = []
        samples_per_class = target_size // 100  # 100 classes in mini-ImageNet
        
        for class_idx in range(100):
            class_indices = indices_by_class[class_idx]
            # Randomly sample from this class
            sampled = np.random.choice(
                class_indices,
                size=min(samples_per_class, len(class_indices)),
                replace=False,
            )
            subset_indices.extend(sampled)
        
        # In case of rounding issues, sample additional indices to reach target_size
        if len(subset_indices) < target_size:
            remaining = target_size - len(subset_indices)
            all_indices = set(range(num_samples))
            available = list(all_indices - set(subset_indices))
            additional = np.random.choice(available, size=remaining, replace=False)
            subset_indices.extend(additional)
        
        return sorted(subset_indices[:target_size])
    
    def __len__(self):
        """Return number of samples in the subset."""
        return len(self.indices)
    
    def __getitem__(self, idx):
        """
        Get a single sample from the dataset.
        
        Args:
            idx (int): Index in the subset
        
        Returns:
            tuple: (image, label) where image is transformed to tensor if transform is applied
        """
        # Map subset index to full dataset index
        full_idx = self.indices[idx]
        
        # Get sample from full dataset
        sample = self.full_dataset[full_idx]
        image = sample["image"]
        label = sample["label"]
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        return image, label
