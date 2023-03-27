

#   This file contains some GPT style models
#   and various presets.


import torch
import torch.nn as nn

from .misc import build_module_registry, get_params_group
from timm.models.registry import register_model


# Custom GPT with maximum flexibility
@register_model
def custom_gpt(
    *args, **kwargs
):
    return GPT(*args, **kwargs)


# GPT-Nano Presets
@register_model
def gpt_nano(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = GPT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        d_model=192, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=384,
        num_layers=3,
    )
    return model


# GPT-Tiny Presets
@register_model
def gpt_tiny(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = GPT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        d_model=256, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=512,
        num_layers=6,
    )
    return model


# GPT-Small Presets
@register_model
def gpt_small(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = GPT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        d_model=384, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=768,
        num_layers=6,
    )
    return model



# GPT-Medium Presets
@register_model
def gpt_medium(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = GPT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        d_model=512, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=1024,
        num_layers=8,
    )
    return model


# GPT-Large Presets
@register_model
def gpt_large(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = GPT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        d_model=768, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=1536,
        num_layers=8,
    )
    return model


# GPT-Extra-Large Presets
@register_model
def gpt_xlarge(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = GPT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        d_model=1024, 
        nhead=8, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=2048,
        num_layers=10,
    )
    return model


# GPT-Huge Presets
@register_model
def gpt_huge(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = GPT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        d_model=1280, 
        nhead=16, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=2560,
        num_layers=12,
    )
    return model


# GPT-Gigantic Presets
@register_model
def gpt_gigantic(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    attn_drop: float = 0.0,
    **kwargs
):
    model = GPT(
        input_dim=input_dim, 
        output_dim=output_dim, 
        d_model=1280, 
        nhead=16, 
        dropout=dropout, 
        attn_drop=attn_drop,
        ff_dim=2560,
        num_layers=24,
    )
    return model


# Default model settings
_default_cfg = {
    "layerscale": {
        "class": "LayerScale",
    },
    "ff_layer": {
        "class": "nn.Linear",
    },
    "activation": {
        "class": "nn.GELU",
    },
    "dropout": {
        "class": "nn.Dropout",
    },
    "layernorm": {
        "class": "nn.LayerNorm",
    },
    "multihead_attn": {
        "class": "nn.MultiheadAttention",
        "params": {
            "batch_first": True,
        }
    },
    "attn_mask": {
        "class": "CasualAttentionMask",
    },
}


class DecoderLayer(nn.Module):
    """
    A single layer of the GPT decoder consisting of self-attention 
    and feedforward layers.
    """

    def __init__(
            self, 
            d_model, 
            nhead: int = 8, 
            dropout: float = 0.0, 
            ff_dim: int = None,
            module_config: dict = {},
            **kwargs,
        ):
        """
        Initializes the DecoderLayer.

        Args:
        - d_model (int): The number of hidden units in the layer.
        - nhead (int): The number of heads in the multi-head attention layer. 
          Default: 8.
        - dropout (float): The dropout rate to apply. Default: 0.1.
        """
        super(DecoderLayer, self).__init__()
        
        # register variables
        self.d_model = d_model
        if ff_dim is None:
            ff_dim = 2 * d_model
            self.ff_dim = ff_dim
            
        # init module registry
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        LayerScale = self.module_registry["layerscale"]
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]
        Dropout = self.module_registry["dropout"]
        LayerNorm = self.module_registry["layernorm"]
        MultiheadAttention = self.module_registry["multihead_attn"]

        # self-attention layer
        self.self_attn = MultiheadAttention(d_model, nhead)
        self.norm1 = LayerNorm(d_model)
        self.scale1 = LayerScale(d_model=d_model)

        # feedforward layer
        self.ff = nn.Sequential(
            FeedForwardLayer(d_model, ff_dim),
            Activation(),
            FeedForwardLayer(ff_dim, d_model),
            Dropout(dropout),
        )
        self.norm2 = LayerNorm(d_model)
        self.scale2 = LayerScale(d_model=d_model)

    def forward(self, x, mask=None, target=None):
        """
        Passes the input through the DecoderLayer.

        Args:
        - x (torch.Tensor): The input tensor of shape 
          (batch_size, sequence_length, d_model).
        - mask (torch.Tensor): An optional mask tensor to apply to 
          the self-attention layer. Default: None.

        Returns:
        - torch.Tensor: The output tensor of shape 
          (batch_size, sequence_length, d_model).
        """
        # if no target, then do self-attention
        if target is None:
            target = x

        # self attention
        residual = target
        x = self.norm1(x)
        x = target + self.scale1(
            self.self_attn(target, x, x, attn_mask=mask)[0]
        )

        # feedforward
        x = self.norm2(x)
        x = x + self.scale2(self.ff(x))

        # residual connection
        x = x + residual
        return x
    
    def forward_next(self, x, mask=None):
        """
        Next token prediction.

        Args:
        - x (torch.Tensor): The input tensor of shape 
          (batch_size, sequence_length, d_model).
        - mask (torch.Tensor): An optional mask tensor to apply to 
          the self-attention layer. Default: None.

        Returns:
        - torch.Tensor: The output tensor of shape 
          (batch_size, sequence_length, d_model).
        """
        return self.forward(x=x, mask=mask, target=x[:, -1:])


    
class GPT(nn.Module):
    """
    The GPT model with compatibility to float output for regression.
    """
    def __init__(
            self, 
            output_dim: int, 
            input_dim: int, 
            d_model: int = 512, 
            nhead: int = 8, 
            dropout: float = 0.0, 
            attn_drop: float = 0.0,
            ff_dim: int = None,
            num_layers: int = 3,
            module_config: dict = {},
            **kwargs,
    ):
        """
        Initializes the GPT model.
        """
        super(GPT, self).__init__()

        # init module registry
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]
        LayerNorm = self.module_registry["layernorm"]
        AttentionMask = self.module_registry["attn_mask"]

        # embedding layer
        self.embedding = FeedForwardLayer(input_dim, d_model)

        # attention mask
        self.attn_mask = AttentionMask(dropout=attn_drop)

        # decoder layers
        decoder_config = module_config.get("decoder_layer", {})
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(
                    d_model=d_model,
                    nhead=nhead,
                    dropout=dropout,
                    ff_dim=ff_dim,
                    module_config=decoder_config,
                )
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        
        # output layer
        self.fc = nn.Sequential(
            FeedForwardLayer(d_model, d_model),
            Activation(),
            FeedForwardLayer(d_model, output_dim),
        )
        self.norm = LayerNorm(d_model)

        # initialize buffer for efficient inference
        self._init_buffer_()
    
    def _init_buffer_(self):
        """ 
        Initialize buffer for efficient next token prediction. 
        """
        self.buffer = [None for _ in range(self.num_layers + 3)]
    
    def forward(self, x):
        """
        Passes the input through the GPT float model.

        Args:
        - x (torch.Tensor): The input tensor of shape 
          (batch_size, sequence_length, input_dim).

        Returns:
        - torch.Tensor: The output tensor of shape 
          (batch_size, sequence_length, output_dim).
        """
        # embedding layer
        out = self.embedding(x)

        # generate attention mask
        attn_mask = self.attn_mask(x.size(1)).to(x.device)

        # decoder blocks
        for block in self.decoder:
            out = block(out, mask=attn_mask)

        # output layer
        out = self.norm(out)
        out = self.fc(out)
        return out
    
    def _forward_last_with_buffer_(self, x, func, buff_idx):
        """
        Forward pass through a layer with a buffer to store the last output.

        Args:
        - x (torch.Tensor): The input tensor of shape 
          (batch_size, sequence_length, input_dim).
        - func (callable): A callable function that takes input tensor `x` and
          returns an output tensor of shape (batch_size, sequence_length, output_dim).
        - buff_idx (int): The index of the buffer to use for storing the last output.

        Returns:
        - torch.Tensor: The output tensor of shape (batch_size, sequence_length, output_dim).
        """
        # embedding layer
        if self.buffer[buff_idx] is None:
            # If buffer is empty, compute the output and store it in buffer
            out = func(x)
            self.buffer[buff_idx] = out.detach()
        else:
            # If buffer is not empty, concatenate the last output with the new output
            # and store the concatenated output in the buffer
            out = torch.cat([
                self.buffer[buff_idx], 
                func(x[:, -1:])
            ], dim=1)
            self.buffer[buff_idx] = out.detach()
        return out
    
    def predict_next(self, x):
        """
        Efficient next token prediction using buffer.]
        WARNING: the buffer will have the memory of last time running this
                 function. If you want a fresh restart, please run 
                 `self._init_buffer_()` before running this function.

        Args:
        - x (torch.Tensor): The input tensor of shape 
                            (batch_size, sequence_length, input_dim).

        Returns:
        - torch.Tensor: The output tensor of shape 
                        (batch_size, sequence_length, output_dim).
        """
        # embedding layer
        out = self._forward_last_with_buffer_(
            x=x, 
            func=self.embedding, 
            buff_idx=0
        )
        
        # self attention mask
        attn_mask = self.attn_mask(x.size(1)).to(x.device)

        # decoder blocks
        for i, block in enumerate(self.decoder):
            buff_idx = i + 1
            if self.buffer[buff_idx] is None:
                out = block(out, mask=attn_mask)
                self.buffer[buff_idx] = out.detach()
            else:
                out = torch.cat([
                    self.buffer[buff_idx], 
                    block.forward_next(out, mask=attn_mask[-1:])
                ], dim=1)
                self.buffer[buff_idx] = out.detach()
            
        # output layer
        out = self._forward_last_with_buffer_(
            x=out, 
            func=self.norm.forward, 
            buff_idx=-2
        )
        out = self._forward_last_with_buffer_(
            x=out, 
            func=self.fc.forward, 
            buff_idx=-1
        )
        return out[:, -1:]
    
    @torch.jit.ignore
    def get_params_group(self, lr=1e-3, weight_decay=1e-4, **kwargs):
        """
        Get the optimizer parameters for training the model.

        Args:
            lr (float): Learning rate for the optimizer. Defaults to 1e-3.
                        weight_decay (float): Weight decay for the optimizer. 
                        Defaults to 1e-4.

        Returns:
            list: A list of dictionaries, where each dictionary specifies 
                  the parameters and optimizer settings for a different parameter group.
        """
        # define the parameter groups for the optimizer
        params_group = [{
            "params": self.parameters(), 
            "lr": float(lr), 
            "weight_decay": float(weight_decay)
        }]
        return params_group
   

