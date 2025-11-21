import torch
import argparse
import matplotlib.pyplot as plt
from src.config import Config
from src.models import Generator

def generate_digit(digit, checkpoint_path, model_type='vanilla', output_path='generated_digit.png'):
    # 1. Initialize Config and Model
    config = Config(model_type=model_type)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    generator = Generator(config).to(device)
    
    # 2. Load Checkpoint
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Handle both full checkpoint (with optimizer) and just state_dict
    if 'state_dict' in checkpoint:
        generator.load_state_dict(checkpoint['state_dict'])
    else:
        generator.load_state_dict(checkpoint)
        
    generator.eval()
    
    # 3. Prepare Input
    # z: Random noise vector (1, LATENT_DIM)
    z = torch.randn(1, config.LATENT_DIM).to(device)
    
    # label: The specific digit we want (Tensor of shape (1,))
    label = torch.tensor([digit]).to(device)
    
    # 4. Generate Image
    with torch.no_grad():
        gen_img = generator(z, label)
    
    # 5. Post-process (Denormalize from [-1, 1] to [0, 1])
    gen_img = gen_img.squeeze().cpu() # Remove batch and channel dims -> (32, 32)
    gen_img = gen_img * 0.5 + 0.5
    
    # 6. Save/Plot
    plt.imshow(gen_img, cmap='gray')
    plt.title(f"Generated Digit: {digit}")
    plt.axis('off')
    plt.savefig(output_path)
    print(f"Image saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a specific digit using a trained GAN")
    parser.add_argument('--digit', type=int, required=True, help="The digit to generate (0-9)")
    parser.add_argument('--checkpoint', type=str, required=True, help="Path to the model checkpoint")
    parser.add_argument('--type', type=str, default='vanilla', choices=['vanilla', 'wgan'], help="Model type")
    parser.add_argument('--out', type=str, default='generated_digit.png', help="Output filename")
    
    args = parser.parse_args()
    
    generate_digit(args.digit, args.checkpoint, args.type, args.out)
