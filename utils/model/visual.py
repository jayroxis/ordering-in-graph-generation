import math
import torch
import torch.nn as nn
import timm


class VisualEncoder(nn.Module):
    def __init__(
        self, 
        model_name: str = "efficientnet_b0", 
        image_size: int = 256, 
        embed_dim: int = 256, 
        num_heads: int = 8, 
        dropout: float = 0.0,
        num_enc_layer = 1,
    ):
        """
        Initialize the visual encoder.

        Args:
            model_name (str, optional): Name of the timm model to use as the encoder. Defaults to "efficientnet_b0".
            image_size (int, optional): The input image size. Defaults to 256.
            embed_dim (int, optional): The hidden size for the transformer layer. Defaults to 256.
            num_heads (int, optional): The number of attention heads for the transformer layer. Defaults to 8.
            dropout (float, optional): The dropout rate for the transformer layer. Defaults to 0.1.
        """
        super(VisualEncoder, self).__init__()
        self.encoder = timm.create_model(model_name, num_classes=0)
        self.output_channels = self.encoder.num_features
        self.image_size = image_size

        # calculate the number of visual tokens
        with torch.no_grad():
            dummy_img = torch.zeros((1, 3, image_size, image_size))
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