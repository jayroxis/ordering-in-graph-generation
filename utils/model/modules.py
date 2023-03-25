

#   This file contains some modules like network blocks
#   and various of useful layers.


import torch
import torch.nn as nn



class LayerScale(nn.Module):
    """
    A scaling layer that scales the output of another 
    layer by a learned scalar value.
    """
    def __init__(self, d_model):
        super(LayerScale, self).__init__()

        self.alpha = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * self.alpha.unsqueeze(0).unsqueeze(0)



class CasualAttentionMask(nn.Module):
    """
    Causal Attention Mask (generation) for GPT Architecture.
    """
    def __init__(self, dropout=0.0, **kwargs):
        super().__init__()
        self.mask = nn.Transformer.generate_square_subsequent_mask
        self.dropout = nn.Dropout(dropout)
        self.device = torch.device("cpu")
        
    def to(self, device, **kwargs):
        """ Rewrite the behavior of `self.to(device)` """
        self.device = device
        super().to(device, **kwargs)

    def forward(self, size):
        """ Generate a causal attention mask. """
        causal_mask = self.mask(size).to(self.device)
        causal_mask = self.dropout(causal_mask)
        return causal_mask