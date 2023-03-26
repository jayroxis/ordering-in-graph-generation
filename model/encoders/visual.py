
import torch
import torch.nn as nn

from model.misc import build_model
from model.misc import build_module_registry

from timm.models.registry import register_model


@register_model
def custom_visual_encoder(
    backbone: str,
    output_dim: int, 
    in_chans: int = 3,
    img_size: int = 256,
    num_heads: int = 8, 
    dropout: float = 0.0,
    transformer_depth = 0, 
    module_config: dict = {},
    **kwargs, 
):
    """
    Visual encoder with maximum flexibility.
    """
    model = VisualEncoder(
        model_name=backbone,
        output_dim=output_dim,
        img_size=img_size,
        module_config=module_config,
        in_chans=in_chans,
        num_heads=num_heads, 
        dropout=dropout,
        transformer_depth=transformer_depth,
        **kwargs, 
    )
    return model


@register_model
def custom_conv_encoder(
    backbone: str,
    output_dim: int, 
    in_chans: int = 3,
    num_heads: int = 8, 
    dropout: float = 0.0,
    transformer_depth = 1,
    module_config: dict = {},
    **kwargs, 
):
    """
    Convolutional encoder with maximum flexibility.
    """
    model = ConvNetEncoder(
        model_name=backbone,
        output_dim=output_dim,
        module_config=module_config,
        in_chans=in_chans,
        num_heads=num_heads, 
        dropout=dropout,
        transformer_depth=transformer_depth,
        **kwargs, 
    )
    return model



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
    "pos_emb": {
        "class": "FourierEmbedding",
    },
}


class VisualEncoder(nn.Module):
    def __init__(
        self, 
        model_name: str,
        output_dim: int, 
        in_chans: int = 3,
        img_size: int = 256, 
        num_heads: int = 8, 
        dropout: float = 0.0,
        transformer_depth = 1,
        module_config: dict = {}, 
        **kwargs,
    ):
        super(VisualEncoder, self).__init__()

        self.in_chans = in_chans
        self.img_size = img_size
        self.output_dim = output_dim

        # create model using timm
        self.encoder = build_model(model_name, num_classes=0, **kwargs)

        # init module registry
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]
        PositionalEmbedding = self.module_registry["pos_emb"]

        # get visual embedding shape
        self.get_encoded_feature_shape()

        # initialize the positional embeddings:
        self.pos_embed = PositionalEmbedding(
            num_tokens=self.num_tokens, 
            d_model=self.output_channels,
        )
        
        # output layer
        self.fc = FeedForwardLayer(self.output_channels, output_dim)
        if transformer_depth < 1:
            self.transformer_encoder = nn.Identity()
        else:
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

    @torch.no_grad()
    def get_encoded_feature_shape(self):
        # calculate the number of visual tokens
        dummy_img = torch.zeros((1, self.in_chans, self.img_size, self.img_size))
        features = self.encoder.forward_features(dummy_img)
        if isinstance(features, list):
            features = features[-1]
        if features.ndim == 4:
            features = features.permute(0, 2, 3, 1)
            features = features.contiguous()
            num_tokens = features.size(1) * features.size(2)
        else:
            num_tokens = features.shape[1]

        self.output_channels = features.size(-1)
        self.num_tokens = num_tokens

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the visual encoder.

        Args:
            x (torch.Tensor): Input image tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Encoded feature tensor with shape (batch_size, num_tokens, output_channels).
        """
        features = self.encoder.forward_features(x)
        if isinstance(features, list):
            features = features[-1]
        if features.ndim == 4:
            features = features.permute(0, 2, 3, 1)
            features = features.contiguous()
            features = features.view(features.size(0), -1, self.output_channels)
        tokens = features + self.pos_embed(features)
        tokens = self.fc(tokens)
        return tokens
    
    @torch.jit.ignore
    def get_params_group(self, lr=1e-3, weight_decay=1e-4):
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
        params = [
            {"params": self.encoder.parameters(), "lr": lr, "weight_decay": weight_decay},
            {"params": self.pos_embed.parameters(), "lr": lr, "weight_decay": 0},
        ]
        return params
    



class ConvNetEncoder(VisualEncoder):
    def __init__(
        self, 
        model_name: str,
        output_dim: int, 
        in_chans: int = 3,
        num_heads: int = 8, 
        dropout: float = 0.0,
        transformer_depth = 1,
        module_config: dict = {}, 
        **kwargs,
    ):
        """
        Visual encoder with Convolutional Backbone.
        The biggest difference is that this module can handle input image of different size.
        """
        super(VisualEncoder, self).__init__()

        self.output_dim = output_dim
        self.in_chans = in_chans

        # create model using timm
        self.encoder = build_model(
            model_name, 
            in_chans=in_chans, 
            num_classes=0, 
            **kwargs
        )

        # init module registry
        if "pos_emb" not in module_config:
            module_config["pos_emb"] ={
                "class": "FourierEncoderPermute2D",
            }
        self.module_registry = build_module_registry(
            config=module_config,
            default_cfg=_default_cfg,
        )
        FeedForwardLayer = self.module_registry["ff_layer"]
        Activation = self.module_registry["activation"]
        PositionalEmbedding = self.module_registry["pos_emb"]

        # get visual embedding shape
        self.get_encoded_feature_shape()

        # initialize the positional embeddings:
        self.pos_embed = PositionalEmbedding(
            channels=self.output_channels,
        )
        
        # output layer
        self.fc = FeedForwardLayer(self.output_channels, output_dim)
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

    @torch.no_grad()
    def get_encoded_feature_shape(self):
        # calculate the number of visual tokens
        dummy_img = torch.zeros((1, self.in_chans, 256, 256))
        features = self.encoder.forward_features(dummy_img)
        if isinstance(features, list):
            features = features[-1]
        if features.ndim == 4:
            features = features.permute(0, 2, 3, 1)
            features = features.contiguous()
        self.output_channels = features.size(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the visual encoder.

        Args:
            x (torch.Tensor): Input image tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Encoded feature tensor with shape (batch_size, num_tokens, output_channels).
        """
        features = self.encoder.forward_features(x)
        tokens = features + self.pos_embed(features)

        tokens = tokens.permute(0, 2, 3, 1)
        tokens = tokens.contiguous()
        tokens = tokens.view(features.size(0), -1, self.output_channels)
        
        tokens = self.fc(tokens)
        tokens = self.transformer_encoder(tokens)
        return tokens