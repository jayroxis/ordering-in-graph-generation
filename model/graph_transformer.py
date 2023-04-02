

import torch
import torch.nn as nn
import torch.nn.functional as F

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
    "query_token": {
        "class": "GaussianEmbedding",
    },
}


@register_model
class GraphTransformer(nn.Module):
    """
    Graph Transformer For Edge-based Tokenization.
    """
    def __init__(
        self, 
        input_dim: int,
        output_dim: int, 
        d_model: int = 512,
        num_heads: int = 8, 
        dropout: float = 0.0,
        enc_depth: int = 3,
        dec_depth: int = 3,
        max_set_size: int = 100,
        module_config: dict = {}, 
        **kwargs,
    ):
        super(GraphTransformer, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.max_set_size = max_set_size
        
        # init module registry
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]
        QueryTokens = self.module_registry["query_token"]

        # initialize the positional embeddings:
        self.query_tokens = QueryTokens(
            num_tokens=max_set_size, 
            d_model=d_model, 
            trainable=True,
        )
        
        # output layer
        self.in_proj = FeedForwardLayer(input_dim, d_model)
        self.fc = FeedForwardLayer(d_model, output_dim)
        self.transformer = nn.Transformer(
            d_model=d_model,
            dim_feedforward=d_model * 2,
            num_encoder_layers=enc_depth,
            num_decoder_layers=dec_depth,
            nhead=num_heads,
            dropout=dropout,
            activation=Activation(),
            batch_first=True,
            norm_first=True
        )

    def forward(self, x, query = None) -> torch.Tensor:
        """
        Main forward function.

        x (torch.Tensor): Edge-based tokens.
        Return:
            output (torch.Tensor): predicted set.
        """
        features = self.in_proj(x)
        if query is None:
            query = self.query_tokens()
            query = query.repeat(len(features), 1, 1)
        output = self.transformer(src=features, tgt=query)
        output = self.fc(output)
        return output
  
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
                "params": self.transformer.parameters(), 
                "lr": lr, 
                "weight_decay": weight_decay, 
                **kwargs
            },
            {
                "params": self.in_proj.parameters(), 
                "lr": lr, 
                "weight_decay": weight_decay, 
                **kwargs
            },
            {
                "params": self.fc.parameters(), 
                "lr": lr, 
                "weight_decay": weight_decay, 
                **kwargs
            },
            {
                # Query tokens will not be regularized.
                "params": self.query_tokens.parameters(), 
                "lr": lr, 
                "weight_decay": 0, 
                **kwargs
            },
        ]
        return params