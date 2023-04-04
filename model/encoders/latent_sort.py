import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.criterion import *

from timm.models.registry import register_model


@register_model
class LatentSortEncoderMLP(nn.Module):
    """
    Latent Sort encoder based on MLP architecture.
    
    This module uses Hungarian algorithm to enable end-to-end latent
    sort training to assist sequential models to learn unordered set.
    """
    def __init__(
        self, 
        pairwise_dist,
        input_dim=16, 
        output_dim=1, 
        hidden_dim=256, 
        num_layers=3,
        norm_out=True,
    ):
        super().__init__()
        
        if type(pairwise_dist) == str:
            self.pairwise_dist = eval(pairwise_dist)
        elif type(pairwise_dist) == dict:
            params = pairwise_dist.get("params", {})
            self.pairwise_dist = eval(pairwise_dist["class"])(**params)
        else:
            self.pairwise_dist = pairwise_dist
            
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.norm_out = norm_out
        
        self.mlp = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.Tanh(),
            nn.LayerNorm(self.hidden_dim),
            *[nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.Tanh(),
                nn.LayerNorm(self.hidden_dim),
            ) for _ in range(num_layers - 1)],
            nn.Linear(self.hidden_dim, self.output_dim)
        )
            
    def forward(self, x):
        latent = self.mlp(x)
        if self.norm_out:
            latent = (latent - latent.mean()) / (latent.std() + 1e-9)
        return latent
    
    def sort(self, x):
        orig_shape = tuple(x.shape)
        if x.ndim == 2:
            x = x.unsqueeze(0)
        latent = self.mlp(x)
        idx = torch.argsort(latent, dim=1).flatten()
        x = x[:, idx].reshape(*orig_shape)
        return x
    
    def loss(self, a, b):
        # Handle single 2D input rather than batches
        if a.ndim == 2:
            a = a.unsqueeze(0)
        if b.ndim == 2:
            b = b.unsqueeze(0)  
        
        # Pairwise cost matrix
        cost = self.pairwise_dist(a, b)
        
        # Matching using Hungarian algorithm
        a_idx, b_idx = [], []
        for c in cost:
            idx = linear_sum_assignment(c)
            a_idx.append(idx[0])
            b_idx.append(idx[1])
        a_idx = torch.from_numpy(np.stack(a_idx)).unsqueeze(-1)
        b_idx = torch.from_numpy(np.stack(b_idx)).unsqueeze(-1)
        
        # Loss for making `a` reordering towards `b`
        latent = self.forward(a)
        latent_idx = torch.argsort(latent, dim=1)
        latent_a = torch.gather(latent, dim=1, index=a_idx)
        loss_a = F.smooth_l1_loss(latent, latent_a)
        
        # Loss for making `b` reordering towards `a`
        latent = self.forward(b)
        latent_idx = torch.argsort(latent, dim=1)
        latent_b = torch.gather(latent, dim=1, index=b_idx)
        loss_b = F.smooth_l1_loss(latent, latent_b)
        
        # Symmetric loss design
        loss = loss_a + loss_b
        return loss