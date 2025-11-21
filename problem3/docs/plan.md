# Wasserstein GAN vs Vanilla GAN (Conditional DCGAN)

## Problem Statement
*   **Objective**: Implement and compare two Generative Adversarial Networks (GANs) on the MNIST dataset.
*   **Architecture**: Use a **Conditional DCGAN** (Deep Convolutional GAN) architecture for both models. This combines the convolutional structure of DCGAN with the class-conditional inputs of CGAN.
*   **Variants**:
    1.  **Vanilla GAN**: Trained with the standard Minimax loss (Binary Cross Entropy).
    2.  **Wasserstein GAN (WGAN)**: Trained with the Wasserstein loss and weight clipping.
*   **Deliverables**:
    *   Compare training stability and convergence.
    *   Provide sample generated images for specific classes.
    *   Discuss the training process and results.

## Project Structure
The project is modularized to separate configuration, data, modeling, and training logic.

```text
problem3/
├── data/                        # Raw MNIST data
├── src/                         # Source code
│   ├── __init__.py
│   ├── config.py                # Hyperparameters & Configuration
│   ├── dataset.py               # MNIST DataLoader & Preprocessing
│   ├── models.py                # Conditional Generator & Discriminator (DCGAN)
│   ├── losses.py                # Loss functions (BCE vs Wasserstein)
│   ├── trainer.py               # Training loops (Vanilla vs WGAN steps)
│   └── utils.py                 # Checkpointing, Logging, Grid generation
├── notebooks/                   # Analysis & Visualization
│   └── visualization.ipynb      # Plotting loss curves & generating samples
├── train.py                     # Main entry point script
└── plan.md                      # This plan
```

## Implementation Plan

### Phase 1: Setup & Data
1. [x] **Project Scaffold**: Create the directory structure and empty `__init__.py` files.
2. [x] **Configuration**: Implement `src/config.py` to hold hyperparameters (batch size, learning rate, latent dim, image size) and a toggle for `model_type` ('vanilla' or 'wgan').
3. [x] **Data Loading**: Implement `src/dataset.py`.
    *   Use `torchvision.datasets.MNIST`.
    *   Resize images to 32x32 or 64x64 (powers of 2 work best for DCGAN).
    *   Normalize to $[-1, 1]$.

### Phase 2: Model Architecture
4.  [x] **Models**: Implement `src/models.py`.
    *   **Generator**: Input $(z, label)$. Use Transposed Convolutions + BatchNorm + ReLU. Output `tanh`.
    *   **Discriminator**: Input $(image, label)$. Use Strided Convolutions + BatchNorm + LeakyReLU.
    *   Ensure conditioning is handled (e.g., by concatenating label embeddings to the input or feature maps).

### Phase 3: Training Logic
5.  [x] **Losses**: Implement `src/losses.py`.
    *   Vanilla: `nn.BCELoss`.
    *   WGAN: Custom functions for $-D(x)$ and $D(G(z))$.
6.  [x] **Trainer**: Implement `src/trainer.py`.
    *   Create a `Trainer` class.
    *   Implement `train_epoch_vanilla()`: Standard 1 discriminator step / 1 generator step.
    *   Implement `train_epoch_wgan()`: $n_{critic}$ discriminator steps per generator step, weight clipping for discriminator.
7.  [x] **Main Script**: Implement `train.py`.
    *   Parse args, setup logger, initialize model/optimizer, run the training loop, save checkpoints.

### Phase 4: Evaluation
8.  [x] **Visualization**: Create `notebooks/visualization.ipynb`.
    *   Load trained weights.
    *   Generate grid of images (e.g., rows=classes, cols=random samples).
    *   Plot loss curves from training logs.