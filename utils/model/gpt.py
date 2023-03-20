import torch
import torch.nn as nn
from .layerscale import LayerScale


# alias for causal mask
casual_mask = nn.Transformer.generate_square_subsequent_mask




class DecoderLayer(nn.Module):
    """
    A single layer of the GPT decoder consisting of self-attention 
    and feedforward layers.
    """

    def __init__(self, d_model, nhead=8, dropout=0.0, batch_first=True):
        """
        Initializes the DecoderLayer.

        Args:
        - d_model (int): The number of hidden units in the layer.
        - nhead (int): The number of heads in the multi-head attention layer. 
          Default: 8.
        - dropout (float): The dropout rate to apply. Default: 0.1.
        - batch_first (bool): If True, expects the batch size to be the 
          first dimension of input tensors. Default: True.
        """
        super(DecoderLayer, self).__init__()

        # self-attention layer
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, batch_first=batch_first
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.scale1 = LayerScale(d_model)

        # feedforward layer
        self.ff = nn.Sequential(
            nn.Linear(d_model, 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.scale2 = LayerScale(d_model)

    def forward(self, x, mask=None):
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
        # self attention
        residual = x
        x = self.norm1(x)
        x = x + self.scale1(
            self.self_attn(x, x, x, attn_mask=mask)[0]
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
        # self attention
        residual = x[:, -1:]
        
        x = self.norm1(x)
        x = x[:, -1:] + self.scale1(
            self.self_attn(x[:, -1:], x, x, attn_mask=mask)[0]
        )

        # feedforward
        x = self.norm2(x)
        x = x + self.scale2(self.ff(x))

        # residual connection
        x = x + residual
        return x



    
class GPT(nn.Module):
    """
    The GPT Float model.
    """

    def __init__(self, output_size, input_size, d_model, num_layers):
        """
        Initializes the GPT Float model.

        Args:
        - output_size (int): The size of the vocabulary.
        - input_size (int): The size of the embedding layer.
        - d_model (int): The number of hidden units in each layer.
        - num_layers (int): The number of layers in the decoder.
        """
        super(GPT, self).__init__()

        # embedding layer
        self.embedding = nn.Linear(input_size, d_model)

        # decoder layers
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(d_model, batch_first=True)
                for _ in range(num_layers)
            ]
        )
        self.num_layers = num_layers
        
        # output layer
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_size),
        )
        self.norm = nn.LayerNorm(d_model)
        self._init_buffer_()
    
    def _init_buffer_(self):
        """ Initialize buffer for efficient next token prediction. """
        self.buffer = [None for _ in range(self.num_layers + 3)]
    
    def forward(self, x):
        """
        Passes the input through the GPT float model.

        Args:
        - x (torch.Tensor): The input tensor of shape 
          (batch_size, sequence_length, input_size).

        Returns:
        - torch.Tensor: The output tensor of shape 
          (batch_size, sequence_length, output_size).
        """
        # embedding layer
        out = self.embedding(x)

        # self attention mask
        attn_mask = casual_mask(x.size(1)).to(x.device)

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
          (batch_size, sequence_length, input_size).
        - func (callable): A callable function that takes input tensor `x` and
          returns an output tensor of shape (batch_size, sequence_length, output_size).
        - buff_idx (int): The index of the buffer to use for storing the last output.

        Returns:
        - torch.Tensor: The output tensor of shape (batch_size, sequence_length, output_size).
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
                            (batch_size, sequence_length, input_size).

        Returns:
        - torch.Tensor: The output tensor of shape 
                        (batch_size, sequence_length, output_size).
        """
        # embedding layer
        out = self._forward_last_with_buffer_(
            x=x, 
            func=self.embedding, 
            buff_idx=0
        )
        
        # self attention mask
        attn_mask = casual_mask(out.size(1)).to(out.device)

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
   