

import torch
import torch.nn as nn

from .misc import build_module_registry, get_params_group
from timm.models.registry import register_model



# Custom LSTM with maximum flexibility
@register_model
def custom_lstm(
    *args, **kwargs
):
    return LSTM(*args, **kwargs)


# LSTM-Nano Presets
@register_model
def lstm_nano(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    **kwargs
):
    model = LSTM(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=256, 
        num_layers=2, 
        dropout=dropout, 
    )
    return model



# LSTM-Tiny Presets
@register_model
def lstm_tiny(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    **kwargs
):
    model = LSTM(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=512, 
        num_layers=2, 
        dropout=dropout, 
    )
    return model


# LSTM-Small Presets
@register_model
def lstm_small(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    **kwargs
):
    model = LSTM(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=512, 
        num_layers=4, 
        dropout=dropout, 
    )
    return model


# LSTM-Medium Presets
@register_model
def lstm_medium(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    **kwargs
):
    model = LSTM(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=1024, 
        num_layers=4, 
        dropout=dropout, 
    )
    return model


# LSTM-Large Presets
@register_model
def lstm_large(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    **kwargs
):
    model = LSTM(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=1280, 
        num_layers=6, 
        dropout=dropout, 
    )
    return model


# LSTM-Extra-Large Presets
@register_model
def lstm_xlarge(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    **kwargs
):
    model = LSTM(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=1600, 
        num_layers=6, 
        dropout=dropout, 
    )
    return model



# LSTM-Gigantic Presets
@register_model
def lstm_gigantic(
    input_dim: int, 
    output_dim: int, 
    dropout: float = 0.0, 
    **kwargs
):
    model = LSTM(
        input_dim=input_dim, 
        output_dim=output_dim, 
        hidden_dim=2560, 
        num_layers=6, 
        dropout=dropout, 
    )
    return model



class LSTM(nn.Module):
    """
    The LSTM model with compatibility to float output for regression.
    """
    def __init__(
            self, 
            output_dim: int, 
            input_dim: int, 
            hidden_dim: int = 512, 
            num_layers: int = 3, 
            dropout: float = 0.0, 
            batch_first: bool = True,
            bidirectional: bool = False,
            **kwargs,
    ):
        """
        Initializes the LSTM model.
        """
        super(LSTM, self).__init__()

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_dim, 
            hidden_size=hidden_dim,
            num_layers=num_layers, 
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional

        # output layer
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc = nn.Linear(lstm_output_dim, output_dim)
        
        self._init_buffer_()
        
    def forward(self, x):
        """
        Passes the input through the LSTM float model.

        Args:
        - x (torch.Tensor): The input tensor of shape 
          (batch_size, sequence_length, input_dim).

        Returns:
        - torch.Tensor: The output tensor of shape 
          (batch_size, sequence_length, output_dim).
        """
        # LSTM layers
        out, _ = self.lstm(x)

        # output layer
        out = self.fc(out)
        return out
    
    def _init_buffer_(self):
        """
        Initialize buffer for efficient next token prediction.
        """
        self.buffer = None

    def predict_next(self, x):
        """
        Efficient next token prediction using buffer.
        WARNING: the buffer will have the memory of last time running this
                 function. If you want a fresh restart, please run
                 `self._init_buffer_()` before running this function.

        Args:
        - x (torch.Tensor): The input tensor of shape
                            (batch_size, sequence_length, input_dim).

        Returns:
        - torch.Tensor: The output tensor of shape
                        (batch_size, sequence_length, output_dim).
        """
        if self.buffer is None:
            out, hidden = self.lstm(x)
            self.buffer = hidden
        else:
            out, hidden = self.lstm(x[:, -1:].contiguous(), self.buffer)
            self.buffer = hidden

        out = self.fc(out)
        return out[:, -1:]
    
    @torch.jit.ignore
    def get_params_group(self, lr=1e-3, weight_decay=1e-4, **kwargs):
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
        params_group = [{
            "params": self.parameters(), 
            "lr": float(lr), 
            "weight_decay": float(weight_decay),
            **kwargs
        }]
        return params_group
