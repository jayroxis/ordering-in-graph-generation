

import torch
import torch.nn as nn

from model.misc import build_module_registry
from timm.models.registry import register_model


# Default model settings
_default_cfg = {
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
}


@register_model
class SinusoidalEncoder(nn.Module):
    def __init__(self, d_model, max_freq=10, **kwargs,):
        """
        Initialize the sinusoidal encoder.

        Args:
        - d_model: output dimensionality of the encoder
        - max_freq: maximum frequency used in the sinusoidal transformation
        """
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
    
    def forward(self, x):
        """
        Apply the sinusoidal encoding to the input.

        Args:
        - x: input tensor of shape (batch_size, set_size, D)

        Returns:
        - pos_enc: tensor of shape (batch_size, set_size, d_model) representing the encoded input
        """
        pos_enc = torch.zeros(x.shape[0], x.shape[1], self.d_model)

        # Compute the set of frequencies
        freqs = torch.pow(2, torch.arange(0, self.d_model, 2) / self.d_model * self.max_freq)

        # Apply the sinusoidal transformation for each frequency
        for i in range(self.d_model // 2):
            pos_enc[:, :, 2*i] = torch.sin(x[:, :, 0] * freqs[i])
            pos_enc[:, :, 2*i+1] = torch.cos(x[:, :, 1] * freqs[i])

        # Handle odd output dimensionality
        if self.d_model % 2 == 1:
            pos_enc[:, :, -1] = torch.sin(x[:, :, 2] * freqs[-1])

        return pos_enc


@register_model    
class SinusoidalMLPEncoder(nn.Module):
    def __init__(
        self, 
        d_model, 
        max_freq=10, 
        module_config: dict = {}, 
        **kwargs,
    ):
        """
        Initialize the sinusoidal MLP encoder.

        Args:
        - d_model: output dimensionality of the encoder
        - max_freq: maximum frequency used in the sinusoidal transformation
        """
        super().__init__()
        self.max_freq = max_freq
        self.d_model = d_model

        # init module registry
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]

        self.mlp = nn.Sequential(
            FeedForwardLayer(2 * max_freq, d_model),
            Activation(),
            FeedForwardLayer(d_model, d_model)
        )
        self._init_weights_()
        
        # Compute the set of frequencies
        self.register_buffer(
            "freqs",
            torch.pow(2, torch.arange(0, 2 * self.max_freq, 2) / (2 * self.max_freq))
        )
        
    def _init_weights_(self):
        """ Initialize MLP weights """
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(
                    m.weight, 
                    std=1 / (0.02 * self.max_freq * self.d_model)
                )
    
    def forward(self, x):
        """
        Apply the sinusoidal encoding and MLP transformation to the input.

        Args:
        - x: input tensor of shape (batch_size, set_size, D)

        Returns:
        - output: tensor of shape (batch_size, set_size, d_model) representing the encoded input
        """
        pos_enc = torch.zeros(
            x.shape[0], 
            x.shape[1], 
            2 * self.max_freq,
            device=x.device, 
            dtype=x.dtype
        )

        # Apply the sinusoidal transformation for each frequency
        for i in range(self.max_freq):
            pos_enc[:, :, 2*i] = torch.sin(x[:, :, 0] * self.freqs[i])
            pos_enc[:, :, 2*i+1] = torch.cos(x[:, :, 1] * self.freqs[i])

        # Apply MLP transformation with GELU activation
        output = self.mlp(pos_enc)

        return output


@register_model
class MLPEncoder(nn.Module):
    def __init__(
        self, 
        input_size, 
        d_model, 
        hidden_size=64, 
        num_layers=2,
        module_config={},
        **kwargs,
    ):
        """
        Initialize the MLP encoder.

        Args:
        - input_size: input dimensionality of the encoder
        - d_model: output dimensionality of the encoder
        - hidden_size: size of the hidden layers in the MLP. Defaults to 64.
        - act: activation function to use in the MLP. Defaults to 'gelu'.
        - num_layers: number of layers in the MLP. Defaults to 2.
        """
        super().__init__()

        # init module registry
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]

        self.d_model = d_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # define the MLP layers
        layers = []
        for i in range(num_layers):
            layers.append(FeedForwardLayer(input_size if i == 0 else hidden_size, hidden_size))
            layers.append(Activation())
        layers.append(FeedForwardLayer(hidden_size, d_model))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Apply the MLP transformation to the input.

        Args:
        - x: input tensor of shape (batch_size, set_size, D)

        Returns:
        - output: tensor of shape (batch_size, set_size, d_model) representing the encoded input
        """
        output = self.mlp(x)

        return output
