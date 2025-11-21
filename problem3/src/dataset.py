import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloader(config):
    """
    Creates and returns the MNIST dataloader.
    """
    transform = transforms.Compose([
        transforms.Resize(config.IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]) # Normalize to [-1, 1]
    ])

    # Ensure data directory exists
    os.makedirs(config.DATA_DIR, exist_ok=True)

    dataset = datasets.MNIST(
        root=config.DATA_DIR,
        train=True,
        download=True,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(config.DEVICE == 'cuda')
    )

    return dataloader
