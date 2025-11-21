# Wasserstein GAN vs Vanilla GAN (Conditional DCGAN)

This project implements and compares two Generative Adversarial Network (GAN) variants on the MNIST dataset:
1.  **Vanilla GAN**: Standard GAN with Binary Cross Entropy loss.
2.  **Wasserstein GAN (WGAN)**: GAN with Wasserstein loss and weight clipping.

Both models utilize a **Conditional DCGAN** architecture, allowing for class-conditional image generation (e.g., generating a specific digit).

## Project Structure

The project is organized as follows:

```text
problem3/
├── data/                        # MNIST dataset storage
├── src/                         # Source code package
│   ├── config.py                # Hyperparameters and configuration
│   ├── dataset.py               # Data loading and preprocessing
│   ├── models.py                # Conditional Generator and Discriminator architectures
│   ├── losses.py                # Loss function implementations
│   ├── trainer.py               # Training loop logic
│   └── utils.py                 # Utilities for logging and saving images
├── notebooks/                   # Jupyter notebooks
│   └── visualization.ipynb      # Analysis and visualization of results
├── train.py                     # Main training script
└── README.md                    # Project documentation
```

## Requirements

*   Python 3.x
*   PyTorch
*   Torchvision
*   Matplotlib
*   Jupyter (for notebooks)

## Usage

### Training

To train the models, use the `train.py` script. You can specify the model type (`vanilla` or `wgan`) via the `--type` command-line argument.

```bash
# Train Vanilla GAN
python train.py --type vanilla

# Train WGAN
python train.py --type wgan
```

Hyperparameters such as batch size, learning rate, and number of epochs can be modified in `src/config.py`.

### Visualization

Open `notebooks/visualization.ipynb` to:
1.  Load trained model checkpoints (saved in `checkpoints/`).
2.  Generate sample images for specific digits.
3.  Visualize training loss curves (logs saved in `logs/`).

## Architecture Details

*   **Generator**: 
    *   Input: Noise vector ($z$) + Class Label Embedding.
    *   Architecture: Transposed convolutions (upsampling) with Batch Normalization and ReLU activations.
    *   Output: Tanh activation (pixel values in $[-1, 1]$).
*   **Discriminator (Critic)**: 
    *   Input: Image + Class Label Embedding (concatenated as an extra channel).
    *   Architecture: Strided convolutions (downsampling) with Batch Normalization and LeakyReLU activations.
    *   Output: 
        *   Vanilla: Sigmoid (via `BCEWithLogitsLoss`).
        *   WGAN: Linear (raw score).
*   **Conditioning**: Class labels are embedded and concatenated with the input noise (Generator) or input image features (Discriminator).
