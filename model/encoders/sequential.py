

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
    "pos_emb": {
        "class": "FourierEmbedding",
    },
    "pos_enc": {
        "class": "FourierEncoder1D",
    }
}



@register_model
class SinusoidalEncoder(nn.Module):
    def __init__(
        self, 
        output_dim, 
        max_freq=10, 
        **kwargs
    ):
        """
        Initialize the sinusoidal encoder.

        Args:
        - output_dim: output dimensionality of the encoder
        - max_freq: maximum frequency used in the sinusoidal transformation
        """
        super().__init__()
        self.output_dim = output_dim
        self.max_freq = max_freq
    
    def forward(self, x):
        """
        Apply the sinusoidal encoding to the input.

        Args:
        - x: input tensor of shape (batch_size, set_size, D)

        Returns:
        - pos_enc: tensor of shape (batch_size, set_size, output_dim) 
                   representing the encoded input
        """
        pos_enc = torch.zeros(x.shape[0], x.shape[1], self.output_dim)

        # Compute the set of frequencies
        freqs = torch.pow(2, torch.arange(0, self.output_dim, 2) / self.output_dim * self.max_freq)

        # Apply the sinusoidal transformation for each frequency
        for i in range(self.output_dim // 2):
            pos_enc[:, :, 2*i] = torch.sin(x[:, :, 0] * freqs[i])
            pos_enc[:, :, 2*i+1] = torch.cos(x[:, :, 1] * freqs[i])

        # Handle odd output dimensionality
        if self.output_dim % 2 == 1:
            pos_enc[:, :, -1] = torch.sin(x[:, :, 2] * freqs[-1])

        return pos_enc


@register_model    
class SinusoidalMLPEncoder(nn.Module):
    def __init__(
        self, 
        output_dim, 
        max_freq=10, 
        module_config: dict = {}, 
        **kwargs,
    ):
        """
        Initialize the sinusoidal MLP encoder.

        Args:
        - output_dim: output dimensionality of the encoder
        - max_freq: maximum frequency used in the sinusoidal transformation
        """
        super().__init__()
        self.max_freq = max_freq
        self.output_dim = output_dim

        # init module registry
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]

        self.mlp = nn.Sequential(
            FeedForwardLayer(2 * max_freq, output_dim),
            Activation(),
            FeedForwardLayer(output_dim, output_dim)
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
                    std=1 / (0.02 * self.max_freq * self.output_dim)
                )
    
    def forward(self, x):
        """
        Apply the sinusoidal encoding and MLP transformation to the input.

        Args:
        - x: input tensor of shape (batch_size, set_size, D)

        Returns:
        - output: tensor of shape (batch_size, set_size, output_dim) 
                  representing the encoded input
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
        input_dim, 
        output_dim, 
        hidden_dim=256, 
        num_layers=2,
        module_config={},
        **kwargs,
    ):
        """
        Initialize the MLP encoder.

        Args:
        - input_dim: input dimensionality of the encoder
        - output_dim: output dimensionality of the encoder
        - hidden_dim: size of the hidden layers in the MLP. Defaults to 64.
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

        self.output_dim = output_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # define the MLP layers
        layers = []
        for i in range(num_layers):
            layers.append(FeedForwardLayer(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(Activation())
        layers.append(FeedForwardLayer(hidden_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Apply the MLP transformation to the input.

        Args:
        - x: input tensor of shape (batch_size, set_size, D)

        Returns:
        - output: tensor of shape (batch_size, set_size, output_dim) 
                  representing the encoded input
        """
        output = self.mlp(x)

        return output



@register_model
class TransformerEncoder(nn.Module):
    def __init__(
        self, 
        input_dim: int,
        output_dim: int, 
        d_model: int = 512,
        num_heads: int = 8, 
        dropout: float = 0.0,
        transformer_depth = 1,
        module_config: dict = {}, 
        **kwargs,
    ):
        super(TransformerEncoder, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model

        # init module registry
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]
        PositionalEncoder = self.module_registry["pos_enc"]

        # initialize the positional embeddings:
        self.pos_enc = PositionalEncoder(
            emb_dim=d_model,
        )
        
        # output layer
        self.in_proj = FeedForwardLayer(input_dim, d_model)
        self.fc = FeedForwardLayer(d_model, output_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=output_dim,
                dim_feedforward=output_dim*2,
                nhead=num_heads,
                dropout=dropout,
                activation=Activation(),
                batch_first=True,
                norm_first=True
            ),
            num_layers=transformer_depth,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Main forward function.

        """
        features = self.in_proj(x)
        tokens = features + self.pos_enc(features)
        tokens = self.transformer_encoder(tokens)
        tokens = self.fc(tokens)
        return tokens
    
    @torch.jit.ignore
    def get_params_group(self, lr=1e-3, weight_decay=1e-4, **kwargs):
        """
        Get the optimizer parameters for training the model.

        Args:
            lr (float): Learning rate for the optimizer. Defaults to 1e-3.
                        weight_decay (float): Weight decay for the optimizer. Defaults to 1e-4.

        Returns:
            list: A list of dictionaries, where each dictionary specifies the parameters 
                  and optimizer settings for a different parameter group.
        """
        # define the parameter groups for the optimizer
        if hasattr(self, "lr"):
            lr = float(self.lr)
        if hasattr(self, "weight_decay"):
            weight_decay = float(self.weight_decay)

        params = [
            {
                "params": self.parameters(), 
                "lr": lr, 
                "weight_decay": weight_decay, 
                **kwargs
            },
        ]
        return params