import torch
from src.config import Config
from src.models import Generator, Discriminator

def test_models():
    config = Config()
    G = Generator(config)
    D = Discriminator(config)
    
    z = torch.randn(5, config.LATENT_DIM)
    labels = torch.randint(0, 10, (5,))
    
    print("Testing Generator...")
    gen_imgs = G(z, labels)
    print(f"Generator Output Shape: {gen_imgs.shape}")
    assert gen_imgs.shape == (5, 1, 32, 32)
    
    print("Testing Discriminator...")
    validity = D(gen_imgs, labels)
    print(f"Discriminator Output Shape: {validity.shape}")
    assert validity.shape == (5, 1)
    
    print("Models initialized and forward pass successful.")

if __name__ == "__main__":
    test_models()
