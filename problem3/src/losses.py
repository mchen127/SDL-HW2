import torch
import torch.nn as nn

def get_loss_function(model_type):
    if model_type == 'vanilla':
        # Use BCEWithLogitsLoss because Discriminator outputs raw logits
        return nn.BCEWithLogitsLoss()
    elif model_type == 'wgan':
        return WGANLoss()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

class WGANLoss:
    def __init__(self):
        pass
        
    def critic_loss(self, real_validity, fake_validity):
        """
        Critic minimizes: E[fake] - E[real]
        """
        return torch.mean(fake_validity) - torch.mean(real_validity)
    
    def generator_loss(self, fake_validity):
        """
        Generator minimizes: -E[fake]
        """
        return -torch.mean(fake_validity)

