import torch
import torch.nn as nn


from timm.models.registry import register_model


@register_model
class SinusoidalEncoder(nn.Module):
    def __init__(self, d_model, max_freq=10):
        """
        Initialize the sinusoidal encoder.

        Args:
        - d_model: output dimensionality of the encoder
        - max_freq: maximum frequency used in the sinusoidal transformation
        """
        super().__init__()
        self.d_model = d_model
        self.max_freq = max_freq
    
    def forward(self, x):
        """
        Apply the sinusoidal encoding to the input.

        Args:
        - x: input tensor of shape (batch_size, set_size, D)

        Returns:
        - pos_enc: tensor of shape (batch_size, set_size, d_model) representing the encoded input
        """
        pos_enc = torch.zeros(x.shape[0], x.shape[1], self.d_model)

        # Compute the set of frequencies
        freqs = torch.pow(2, torch.arange(0, self.d_model, 2) / self.d_model * self.max_freq)

        # Apply the sinusoidal transformation for each frequency
        for i in range(self.d_model // 2):
            pos_enc[:, :, 2*i] = torch.sin(x[:, :, 0] * freqs[i])
            pos_enc[:, :, 2*i+1] = torch.cos(x[:, :, 1] * freqs[i])

        # Handle odd output dimensionality
        if self.d_model % 2 == 1:
            pos_enc[:, :, -1] = torch.sin(x[:, :, 2] * freqs[-1])

        return pos_enc
    
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
            {"params": self.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]
        return params

    
@register_model    
class SinusoidalMLPEncoder(nn.Module):
    def __init__(self, d_model, max_freq=10):
        """
        Initialize the sinusoidal MLP encoder.

        Args:
        - d_model: output dimensionality of the encoder
        - max_freq: maximum frequency used in the sinusoidal transformation
        """
        super().__init__()
        self.max_freq = max_freq
        self.d_model = d_model
        self.mlp = nn.Sequential(
            nn.Linear(2 * max_freq, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model)
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
                    std=1 / (0.02 * self.max_freq * self.d_model)
                )
    
    def forward(self, x):
        """
        Apply the sinusoidal encoding and MLP transformation to the input.

        Args:
        - x: input tensor of shape (batch_size, set_size, D)

        Returns:
        - output: tensor of shape (batch_size, set_size, d_model) representing the encoded input
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
            {"params": self.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]
        return params
    


@register_model
class MLPEncoder(nn.Module):
    def __init__(
        self, 
        input_size, 
        d_model, 
        hidden_size=64, 
        act='gelu', 
        num_layers=2
    ):
        """
        Initialize the MLP encoder.

        Args:
        - input_size: input dimensionality of the encoder
        - d_model: output dimensionality of the encoder
        - hidden_size: size of the hidden layers in the MLP. Defaults to 64.
        - act: activation function to use in the MLP. Defaults to 'gelu'.
        - num_layers: number of layers in the MLP. Defaults to 2.
        """
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # define the MLP layers
        layers = []
        for i in range(num_layers):
            layers.append(nn.Linear(input_size if i == 0 else hidden_size, hidden_size))
            if act == 'relu':
                layers.append(nn.ReLU())
            elif act == 'leaky_relU':
                layers.append(nn.LeakyReLU())
            elif act == 'tanh':
                layers.append(nn.Tanh())
            elif act == 'sigmoid':
                layers.append(nn.Sigmoid())
            else:
                layers.append(nn.GELU())
        layers.append(nn.Linear(hidden_size, d_model))
        
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        """
        Apply the MLP transformation to the input.

        Args:
        - x: input tensor of shape (batch_size, set_size, D)

        Returns:
        - output: tensor of shape (batch_size, set_size, d_model) representing the encoded input
        """
        output = self.mlp(x)

        return output

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
            {"params": self.parameters(), "lr": lr, "weight_decay": weight_decay},
        ]
        return params

