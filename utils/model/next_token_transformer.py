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
        pad_value=-1,
        num_encoder_layers=2,
        num_decoder_layers=4,
        **kwargs,
    ):
        super().__init__()

        self.d_model = d_model
        self.input_size = input_size
        self.output_size = output_size
        self.pad_value = pad_value

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
        src_mask = (token == self.pad_value).all(-1)
        emb = self.transformer(
            src=token, 
            tgt=next_token, 
            src_key_padding_mask=src_mask
        )
        
        # output projection
        output = self.out_proj(emb)
        return output

    @torch.jit.ignore
    def get_params_group(self, lr=2e-4, weight_decay=1e-4):
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
        params = [
            {"params": self.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]
        return params