import torch
import torch.optim as optim
from src.losses import get_loss_function
from src.utils import save_checkpoint, generate_sample_grid
import os
from torchvision.utils import save_image
import numpy as np

import json

class Trainer:
    def __init__(self, generator, discriminator, dataloader, config):
        self.G = generator.to(config.DEVICE)
        self.D = discriminator.to(config.DEVICE)
        self.dataloader = dataloader
        self.config = config
        
        if config.MODEL_TYPE == 'wgan':
            # WGAN paper recommends RMSprop
            self.g_optimizer = optim.RMSprop(self.G.parameters(), lr=0.00005)
            self.d_optimizer = optim.RMSprop(self.D.parameters(), lr=0.00005)
        else:
            self.g_optimizer = optim.Adam(self.G.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
            self.d_optimizer = optim.Adam(self.D.parameters(), lr=config.LR, betas=(config.BETA1, config.BETA2))
        
        self.criterion = get_loss_function(config.MODEL_TYPE)
        
        # Fixed noise for visualization
        self.fixed_z = torch.randn(100, config.LATENT_DIM).to(config.DEVICE)
        self.fixed_labels = torch.tensor([i for i in range(10) for _ in range(10)]).to(config.DEVICE)
        self.g_losses = []
        self.d_losses = []
        
        # Create directories
        os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        os.makedirs(config.IMG_DIR, exist_ok=True)

    def save_logs(self):
        logs = {
            'g_losses': self.g_losses,
            'd_losses': self.d_losses
        }
        log_path = os.path.join(self.config.LOG_DIR, f"logs_{self.config.MODEL_TYPE}.json")
        with open(log_path, "w") as f:
            json.dump(logs, f)

    def train(self):
        print(f"Starting training for {self.config.EPOCHS} epochs...")
        
        for epoch in range(self.config.EPOCHS):
            if self.config.MODEL_TYPE == 'vanilla':
                self.train_epoch_vanilla(epoch)
            else:
                self.train_epoch_wgan(epoch)
            
            # Save checkpoint every 10 epochs
            if (epoch + 1) % 10 == 0:
                ckpt_path = os.path.join(self.config.CHECKPOINT_DIR, f"checkpoint_{self.config.MODEL_TYPE}_{epoch+1}.pth.tar")
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': self.G.state_dict(),
                    'optimizer': self.g_optimizer.state_dict(),
                }, filename=ckpt_path)
                
                # Generate sample images
                with torch.no_grad():
                    self.G.eval()
                    gen_imgs = self.G(self.fixed_z, self.fixed_labels)
                    gen_imgs = gen_imgs * 0.5 + 0.5
                    img_path = os.path.join(self.config.IMG_DIR, f"{self.config.MODEL_TYPE}_{epoch+1}.png")
                    save_image(gen_imgs, img_path, nrow=10)
                    self.G.train()
            
            # Save logs at the end of each epoch
            self.save_logs()


    def train_epoch_vanilla(self, epoch):
        for i, (imgs, labels) in enumerate(self.dataloader):
            batch_size = imgs.shape[0]
            
            real_imgs = imgs.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            
            # -----------------
            #  Train Generator
            # -----------------
            self.g_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.config.DEVICE)
            gen_labels = torch.randint(0, self.config.NUM_CLASSES, (batch_size,)).to(self.config.DEVICE)
            
            gen_imgs = self.G(z, gen_labels)
            
            validity = self.D(gen_imgs, gen_labels)
            
            # Generator wants Discriminator to say these are real (1)
            # BCEWithLogitsLoss takes logits and targets
            g_loss = self.criterion(validity, torch.ones_like(validity))
            
            g_loss.backward()
            self.g_optimizer.step()
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.d_optimizer.zero_grad()
            
            # Real loss
            real_pred = self.D(real_imgs, labels)
            d_real_loss = self.criterion(real_pred, torch.ones_like(real_pred))
            
            # Fake loss
            fake_pred = self.D(gen_imgs.detach(), gen_labels)
            d_fake_loss = self.criterion(fake_pred, torch.zeros_like(fake_pred))
            
            d_loss = (d_real_loss + d_fake_loss) / 2
            
            d_loss.backward()
            self.d_optimizer.step()
            
            self.g_losses.append(g_loss.item())
            self.d_losses.append(d_loss.item())
            
            if i % 100 == 0:
                print(f"[Epoch {epoch}/{self.config.EPOCHS}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

    def train_epoch_wgan(self, epoch):
        for i, (imgs, labels) in enumerate(self.dataloader):
            batch_size = imgs.shape[0]
            
            real_imgs = imgs.to(self.config.DEVICE)
            labels = labels.to(self.config.DEVICE)
            
            # ---------------------
            #  Train Discriminator
            # ---------------------
            self.d_optimizer.zero_grad()
            
            z = torch.randn(batch_size, self.config.LATENT_DIM).to(self.config.DEVICE)
            gen_labels = torch.randint(0, self.config.NUM_CLASSES, (batch_size,)).to(self.config.DEVICE)
            
            gen_imgs = self.G(z, gen_labels).detach()
            
            # Adversarial loss
            real_validity = self.D(real_imgs, labels)
            fake_validity = self.D(gen_imgs, gen_labels)
            
            d_loss = self.criterion.critic_loss(real_validity, fake_validity)
            
            d_loss.backward()
            self.d_optimizer.step()
            
            # Clip weights of discriminator
            for p in self.D.parameters():
                p.data.clamp_(-self.config.CLIP_VALUE, self.config.CLIP_VALUE)
                
            # Train the generator every n_critic iterations
            if i % self.config.N_CRITIC == 0:
                # -----------------
                #  Train Generator
                # -----------------
                self.g_optimizer.zero_grad()
                
                gen_imgs = self.G(z, gen_labels)
                fake_validity = self.D(gen_imgs, gen_labels)
                
                g_loss = self.criterion.generator_loss(fake_validity)
                
                g_loss.backward()
                self.g_optimizer.step()
                
                self.g_losses.append(g_loss.item())
                self.d_losses.append(d_loss.item())
                
                if i % 100 == 0:
                    print(f"[Epoch {epoch}/{self.config.EPOCHS}] [Batch {i}/{len(self.dataloader)}] [D loss: {d_loss.item()}] [G loss: {g_loss.item()}]")

