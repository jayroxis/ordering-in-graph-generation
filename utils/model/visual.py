import torch
import torch.nn as nn
import timm


class PositionModule(nn.Module):
    def __init__(self):
        """
        Initialize the PositionModule class.
        """
        super(PositionModule, self).__init__()

    def forward(self, x):
        """
        Compute the positions of every pixel in the input tensor x.

        Args:
            x: A PyTorch tensor of shape (B, C, W, H).

        Returns:
            A PyTorch tensor of shape (B, 2, W, H) containing the positions
            of every pixel, normalized between 0 and 1.
        """
        B, C, W, H = x.size()

        # Create x and y coordinate grids using the `torch.linspace()` function
        y_coords = torch.linspace(
            0, 1, W, device=x.device, dtype=x.dtype
        ).view(1, 1, W, 1).expand(B, 1, W, H)

        x_coords = torch.linspace(
            0, 1, H, device=x.device, dtype=x.dtype
        ).view(1, 1, 1, H).expand(B, 1, W, H)

        # Concatenate the x and y coordinates to create the positions tensor
        positions = torch.cat((x_coords, y_coords), dim=1).type(x.dtype)

        return positions
    


class VisualEncoder(nn.Module):
    def __init__(
        self, 
        model_name: str = "efficientnet_b0", 
        img_size: int = 256, 
        embed_dim: int = 256, 
        num_heads: int = 8, 
        dropout: float = 0.0,
        num_enc_layer: int = 1,
        in_chans: int = 3,
        use_pixel_pos_overlay: bool = False,
        **kwargs
    ):
        """
        Initialize the visual encoder.

        Args:
            model_name (str, optional): Name of the timm model to use as the encoder. Defaults to "efficientnet_b0".
            img_size (int, optional): The input image size. Defaults to 256.
            embed_dim (int, optional): The hidden size for the transformer layer. Defaults to 256.
            num_heads (int, optional): The number of attention heads for the transformer layer. Defaults to 8.
            dropout (float, optional): The dropout rate for the transformer layer. Defaults to 0.1.
            in_chans (int, optional): The number of channels in input image. Defaults to 3.
            use_pixel_pos_overlay (bool, optional): Whether to add positional information as an overlay on the input image.
                                                     Defaults to False.
        """
        super(VisualEncoder, self).__init__()

        # Pixel position overlay layer
        self.use_pixel_pos_overlay = use_pixel_pos_overlay
        if use_pixel_pos_overlay:
            self.pixel_overlay = PositionModule()
            in_chans = in_chans + 2

        # Backbone encoder model
        self.encoder = timm.create_model(model_name, num_classes=0, in_chans=in_chans)
        self.output_channels = self.encoder.num_features
        self.img_size = img_size

        # calculate the number of visual tokens
        with torch.no_grad():
            dummy_img = torch.zeros((1, in_chans, img_size, img_size))
            features = self.encoder.forward_features(dummy_img)
            features = features.permute(0, 2, 3, 1)
            features = features.contiguous()
            num_tokens = features.size(1) * features.size(2)
        self.num_tokens = num_tokens

        # initialize the positional embeddings:
        #    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        self.positional_embeddings = nn.Parameter(
            torch.randn(
                1, 
                self.num_tokens, 
                self.output_channels
            ) * 0.02
        )
        
        self.fc = nn.Linear(self.output_channels, embed_dim)
        self.transformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                dim_feedforward=embed_dim*2,
                nhead=num_heads,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True
            ),
            num_layers=num_enc_layer,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass the input through the visual encoder.

        Args:
            x (torch.Tensor): Input image tensor with shape (batch_size, channels, height, width).

        Returns:
            torch.Tensor: Encoded feature tensor with shape (batch_size, num_tokens, output_channels).
        """
        if self.use_pixel_pos_overlay:
            x = torch.cat([x, self.pixel_overlay(x)], dim=1)
        features = self.encoder.forward_features(x)
        features = features.permute(0, 2, 3, 1)
        features = features.contiguous()
        tokens = features.view(features.size(0), -1, self.output_channels)
        tokens = tokens + self.positional_embeddings
        tokens = self.fc(tokens)
        tokens = self.transformer_encoder(tokens)
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
            {"params": self.transformer_encoder.parameters(), "lr": lr, "weight_decay": weight_decay},
            {"params": self.encoder.parameters(), "lr": lr, "weight_decay": weight_decay},
            {"params": self.positional_embeddings, "lr": lr, "weight_decay": 0},
        ]
        return params