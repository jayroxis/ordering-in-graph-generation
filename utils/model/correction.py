
import torch
import torch.nn as nn


class GraphTransformer(nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        d_model, 
        num_layers=3,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.input_size = input_size
        self.output_size = output_size
        self.num_layers = num_layers

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=8, 
            dim_feedforward=2 * d_model, 
            dropout=0.0, 
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.in_proj = nn.Linear(input_size, d_model)
        self.out_proj = nn.Linear(d_model, output_size)

    def forward(self, vis_emb, node_pair):
        assert vis_emb.shape[-1] == self.d_model, \
        f"last dimension of vis_emb is expected to be {self.d_model}," + \
        f" but getting shape {vis_emb.shape}"

        edge_emb = self.in_proj(node_pair)
        token = torch.cat([vis_emb, edge_emb], dim=1)
        emb = self.encoder(token)

        L = node_pair.shape[1]
        output = self.out_proj(emb[:, -L:])
        return output