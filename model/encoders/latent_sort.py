import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from scipy.optimize import linear_sum_assignment

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
        input_dim, 
        output_dim=1, 
        hidden_dim=256, 
        num_layers=3,
        norm_out=True,
        **kwargs,
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
        latent = self.forward(x).squeeze(-1)
        idx = torch.argsort(latent, dim=1)
        for n, i in enumerate(idx):
            x[n] = x[n][i]
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
            c = c.detach().cpu().numpy()
            idx = linear_sum_assignment(c)
            a_idx.append(idx[0])
            b_idx.append(idx[1])
        a_idx = torch.from_numpy(np.stack(a_idx)).unsqueeze(-1)
        b_idx = torch.from_numpy(np.stack(b_idx)).unsqueeze(-1)
        a_idx = a_idx.to(a.device).long()
        b_idx = b_idx.to(b.device).long()

        # Loss for making `a` reordering towards `b`
        latent = self.forward(a)
        latent_a = torch.gather(latent, dim=1, index=a_idx)
        loss_a = F.smooth_l1_loss(latent, latent_a)
        
        # Loss for making `b` reordering towards `a`
        latent = self.forward(b)
        latent_b = torch.gather(latent, dim=1, index=b_idx)
        loss_b = F.smooth_l1_loss(latent, latent_b)
        
        # Symmetric loss design
        loss = loss_a + loss_b
        return loss
    


@register_model
class LatentSortEncoderTransformer(LatentSortEncoderMLP):
    """
    Latent Sort encoder based on Transformer architecture.
    """
    def __init__(
        self, 
        pairwise_dist,
        input_dim, 
        output_dim=1, 
        hidden_dim=256, 
        num_layers=3,
        norm_out=True,
        nhead=4,
        **kwargs,
    ):
        super(LatentSortEncoderMLP, self).__init__()
        
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
        
        self.in_proj = nn.Linear(input_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, output_dim)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=nhead, 
                dim_feedforward=2 * hidden_dim, 
                dropout=0.0, 
                batch_first=True, 
                norm_first=False,
            ),
            num_layers
        )
            
    def forward(self, x):
        x = self.in_proj(x)
        latent = self.transformer(x)
        latent = self.out_proj(latent)
        if self.norm_out:
            latent = (latent - latent.mean()) / (latent.std() + 1e-9)
        return latent