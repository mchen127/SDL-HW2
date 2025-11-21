import torch

class Config:
    # Data
    DATA_DIR = '../data'
    IMG_SIZE = 32  # Resize MNIST from 28x28 to 32x32 for easier convolutions
    CHANNELS = 1
    BATCH_SIZE = 128
    NUM_WORKERS = 64

    # Model Architecture
    LATENT_DIM = 100
    NUM_CLASSES = 10
    EMBED_SIZE = 50 # Dimension of label embedding

    # Training
    EPOCHS = 100
    LR = 0.0002
    BETA1 = 0.5  # For Adam optimizer
    BETA2 = 0.999
    
    # WGAN specific
    N_CRITIC = 5        # Train discriminator n times per generator step
    CLIP_VALUE = 0.01   # Weight clipping for WGAN
    
    # Paths
    CHECKPOINT_DIR = 'checkpoints'
    LOG_DIR = 'logs'
    IMG_DIR = 'images'

    # System
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def __init__(self, model_type='vanilla'):
        assert model_type in ['vanilla', 'wgan'], "model_type must be 'vanilla' or 'wgan'"
        self.MODEL_TYPE = model_type

    def display(self):
        print(f"Config: {self.MODEL_TYPE.upper()} GAN")
        print(f"Device: {self.DEVICE}")
        print("-" * 20)
