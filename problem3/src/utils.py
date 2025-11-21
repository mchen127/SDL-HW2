import torch
import os
from torchvision.utils import make_grid
import matplotlib.pyplot as plt

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    torch.save(state, filename)

def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])

def generate_sample_grid(generator, device, latent_dim, num_classes, n_row=10):
    """
    Generates a grid of images, one row per class.
    """
    generator.eval()
    z = torch.randn(n_row * num_classes, latent_dim).to(device)
    # Create labels: 0,0,0... 1,1,1...
    labels = torch.tensor([num for num in range(num_classes) for _ in range(n_row)]).to(device)
    
    with torch.no_grad():
        gen_imgs = generator(z, labels)
    
    # Denormalize
    gen_imgs = gen_imgs * 0.5 + 0.5
    
    grid = make_grid(gen_imgs, nrow=n_row, normalize=False)
    return grid

def plot_loss(g_losses, d_losses, save_path):
    plt.figure(figsize=(10,5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_losses, label="G")
    plt.plot(d_losses, label="D")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(save_path)
    plt.close()

