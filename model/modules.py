

#   This file contains some modules like network blocks
#   and various of useful layers.


import torch
import torch.nn as nn

from timm.models.registry import register_model


@register_model
def image_patching_16(in_chans=3, emb_dim=1024, **kwargs):
    module = nn.Conv2d(
        in_channels=in_chans,
        out_channels=emb_dim,
        kernel_size=16,
        stride=16,
    )
    module.forward_features = module.forward
    return module


@register_model
class LayerScale(nn.Module):
    """
    A scaling layer that scales the output of another 
    layer by a learned scalar value.
    """
    def __init__(self, d_model: int, *args, **kwargs):
        super(LayerScale, self).__init__()

        self.alpha = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        return x * self.alpha.unsqueeze(0).unsqueeze(0)


@register_model
class CasualAttentionMask(nn.Module):
    """
    Causal Attention Mask (generation) for GPT Architecture.
    """
    def __init__(self, dropout: float = 0.0, *args, **kwargs):
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



@register_model
class StopTokenDetectorFloat(nn.Module):
    """
    Stop token detector for float sequence.
    """
    def __init__(self, stop_value, threshold=0.1, **kwargs):
        super(StopTokenDetectorFloat, self).__init__()
        self.stop_value = stop_value
        self.threshold = threshold

    def forward(self, sequence):
        if sequence.ndim == 3:
            last_tokens = sequence[:, -1, :]
        elif sequence.ndim == 2:
            last_tokens = sequence
        else:
            raise ValueError(
                "The input has to be a 3D sequence (B, L, D)" + \
                " or a single token (B, D)."
            )
        diff = torch.abs(last_tokens - self.stop_value)
        stop_flag = torch.all(diff <= self.threshold)
        return stop_flag


@register_model
class StopTokenDetectorCategorical(nn.Module):
    """
    Stop token detector for float sequence.
    """
    def __init__(self, stop_idx, threshold=0.8, **kwargs):
        super(StopTokenDetectorCategorical, self).__init__()
        if isinstance(stop_idx, int):
            self.stop_idx = [stop_idx]
        elif isinstance(stop_idx, list):
            self.stop_idx = stop_idx
        elif isinstance(stop_idx, str):
            self.stop_idx = stop_idx.strip(" ").split(",")
            self.stop_idx = [int(i) for i in self.stop_idx if i != ""]
        else:
            raise ValueError("stop_idx has to be a single integer or a list of integers.")
        self.threshold = threshold

    def forward(self, sequence):
        if sequence.ndim == 3:
            last_tokens = sequence[:, -1, :]
        elif sequence.ndim == 2:
            last_tokens = sequence
        else:
            raise ValueError(
                "The input has to be a 3D sequence (B, L, D)" + \
                " or a single token (B, D)."
            )
        if (last_tokens.sum(-1) != 1.0).any():
            if (last_tokens > 0).all():
                last_tokens = last_tokens / last_tokens.sum(-1).unsqueeze(-1)
            else:
                last_tokens = last_tokens.softmax(-1)
        above_threshold = last_tokens[..., self.stop_idx] > self.threshold
        return above_threshold.all().item()
    

@register_model
class DropPath(nn.Module):
    def __init__(self, p: float = 0.):
        super().__init__()
        self.p = p

    def forward(self, x):
        if self.p == 0. or not self.training:
            return x
        keep_prob = 1 - self.p
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output