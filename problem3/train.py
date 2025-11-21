import argparse
from src.config import Config
from src.dataset import get_dataloader
from src.models import Generator, Discriminator, initialize_weights
from src.trainer import Trainer

def main():
    parser = argparse.ArgumentParser(description="Train GAN on MNIST")
    parser.add_argument('--type', type=str, default='vanilla', choices=['vanilla', 'wgan'],
                        help='Type of GAN to train: vanilla or wgan')
    args = parser.parse_args()

    # Initialize Config
    config = Config(model_type=args.type)
    config.display()

    # Load Data
    dataloader = get_dataloader(config)
    print(f"Data loaded. Batches: {len(dataloader)}")

    # Initialize Models
    G = Generator(config)
    D = Discriminator(config)
    
    # Initialize weights
    G.apply(initialize_weights)
    D.apply(initialize_weights)
    
    print("Models initialized.")

    # Initialize Trainer
    trainer = Trainer(G, D, dataloader, config)
    
    # Start Training
    trainer.train()

if __name__ == "__main__":
    main()

