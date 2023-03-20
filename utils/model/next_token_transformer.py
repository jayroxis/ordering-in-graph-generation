import torch
import torch.nn as nn


class NextTokenTransformer(nn.Module):
    def __init__(
        self, 
        input_size, 
        output_size, 
        d_model, 
        nhead=8,
        dropout=0.0, 
        num_encoder_layers=2,
        num_decoder_layers=4,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.input_size = input_size
        self.output_size = output_size

        self.transformer = nn.Transformer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=2 * d_model,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dropout=dropout,
            activation="gelu", 
            batch_first=True, 
            norm_first=True
        )
        
        self.in_proj = nn.Linear(input_size, d_model)
        self.out_proj = nn.Linear(d_model, output_size)
        self.next_token = nn.Parameter(torch.randn(1, 1, d_model))

    def forward(self, token):
        assert token.shape[-1] == self.d_model, \
            f"last dimension of input is expected to be {self.d_model}," + \
            f" but getting shape {token.shape}"

        # concat tokens
        token = self.in_proj(token)
        
        # get next token as target token
        next_token = self.next_token.repeat(len(token), 1, 1)
        
        # encoder-decoder
        src_mask = (token == -1).all(-1)
        emb = self.transformer(
            src=token, 
            tgt=next_token, 
            src_key_padding_mask=src_mask
        )
        
        # output projection
        output = self.out_proj(emb)
        return output
