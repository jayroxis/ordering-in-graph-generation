

#   This file contains some modules like network blocks
#   and various of useful layers.


import torch
import torch.nn as nn

from timm.models.registry import register_model


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
def stop_token_dector(dtype: str, *args, **kwargs):
    if dtype == "int" or dtype == "categorical":
        model = StopTokenDetectorCategorical(*args, **kwargs)
    else:
        model = StopTokenDetectorFloat(*args, **kwargs)
    return model


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
    def __init__(self, stop_idx, threshold=0.8):
        super(StopTokenDetectorCategorical, self).__init__()
        self.stop_idx = stop_idx
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
        softmax_probs = nn.functional.softmax(last_tokens, dim=-1)
        predicted_indices = torch.argmax(softmax_probs, dim=-1)
        max_probs, _ = torch.max(softmax_probs, dim=-1)
        stop_flag = torch.all(predicted_indices == self.stop_idx)
        if max_probs >= self.threshold:
            return stop_flag
        else:
            return False
