import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, config):
        super(Generator, self).__init__()
        self.init_size = config.IMG_SIZE // 8  # 32 // 8 = 4
        self.dim = 128 # Base feature map size

        self.label_emb = nn.Embedding(config.NUM_CLASSES, config.EMBED_SIZE)
        
        self.l1 = nn.Sequential(
            nn.Linear(config.LATENT_DIM + config.EMBED_SIZE, self.dim * 4 * self.init_size ** 2)
        )

        self.conv_blocks = nn.Sequential(
            # Input: (dim*4, 4, 4) -> (512, 4, 4)
            nn.BatchNorm2d(self.dim * 4),
            nn.ReLU(True),
            
            # (512, 4, 4) -> (256, 8, 8)
            nn.ConvTranspose2d(self.dim * 4, self.dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 2),
            nn.ReLU(True),
            
            # (256, 8, 8) -> (128, 16, 16)
            nn.ConvTranspose2d(self.dim * 2, self.dim, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim),
            nn.ReLU(True),
            
            # (128, 16, 16) -> (channels, 32, 32)
            nn.ConvTranspose2d(self.dim, config.CHANNELS, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, z, labels):
        # z: (batch, latent_dim)
        # labels: (batch,)
        gen_input = torch.cat((self.label_emb(labels), z), -1)
        out = self.l1(gen_input)
        out = out.view(out.shape[0], self.dim * 4, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()
        self.img_size = config.IMG_SIZE
        
        # Embedding for conditioning: creates a channel of size (img_size, img_size)
        self.label_embedding = nn.Embedding(config.NUM_CLASSES, config.IMG_SIZE * config.IMG_SIZE)
        
        self.dim = 64

        self.model = nn.Sequential(
            # Input: (channels + 1, 32, 32)
            # (2, 32, 32) -> (64, 16, 16)
            nn.Conv2d(config.CHANNELS + 1, self.dim, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (64, 16, 16) -> (128, 8, 8)
            nn.Conv2d(self.dim, self.dim * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 2),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (128, 8, 8) -> (256, 4, 4)
            nn.Conv2d(self.dim * 2, self.dim * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim * 4),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (256, 4, 4) -> (1, 1, 1)
            nn.Conv2d(self.dim * 4, 1, 4, 1, 0, bias=False),
        )
        
        # Note: No Sigmoid here. 
        # Vanilla GAN will use BCEWithLogitsLoss.
        # WGAN will use raw scores.

    def forward(self, img, labels):
        # img: (batch, channels, 32, 32)
        # labels: (batch,)
        
        # Create label channel
        label_emb = self.label_embedding(labels)
        label_emb = label_emb.view(label_emb.shape[0], 1, self.img_size, self.img_size)
        
        # Concatenate
        d_in = torch.cat((img, label_emb), dim=1)
        
        validity = self.model(d_in)
        # Output shape: (batch, 1, 1, 1) -> flatten to (batch, 1)
        return validity.view(-1, 1)

def initialize_weights(m):
    # Initialize weights according to the DCGAN paper
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
