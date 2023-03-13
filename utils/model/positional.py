import torch
import numpy as np


class SinusoidalEncoder(torch.nn.Module):
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

    
    
class SinusoidalMLPEncoder(torch.nn.Module):
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
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(2 * max_freq, d_model),
            torch.nn.GELU(),
            torch.nn.LayerNorm(d_model),
            torch.nn.Linear(d_model, d_model)
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
            if isinstance(m, torch.nn.Linear):
                torch.nn.init.normal_(
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
    


class MLPEncoder(torch.nn.Module):
    def __init__(self, input_size, d_model, hidden_size=64):
        """
        Initialize the MLP encoder.

        Args:
        - input_size: input dimensionality of the encoder
        - d_model: output dimensionality of the encoder
        """
        super().__init__()
        self.d_model = d_model
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(input_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_size),
            torch.nn.Linear(hidden_size, d_model)
        )

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


    
def test_sinusoidal_encoder(
    # Define test input
    batch_size = 2,
    set_size = 3,
    d_model = 8,
    max_freq = 5,
):
    x = torch.rand(batch_size, set_size, d_model)

    # Initialize encoder and compute output
    encoder = SinusoidalEncoder(d_model=d_model, max_freq=max_freq)
    output = encoder(x)

    # Check output shape
    assert output.shape == (batch_size, set_size, d_model)

    # Check output values
    expected_output = torch.zeros(batch_size, set_size, d_model)
    freqs = torch.pow(2, torch.arange(0, d_model, 2) / d_model * max_freq)
    for i in range(d_model // 2):
        expected_output[:, :, 2*i] = torch.sin(x[:, :, 0] * freqs[i])
        expected_output[:, :, 2*i+1] = torch.cos(x[:, :, 1] * freqs[i])
    if d_model % 2 == 1:
        expected_output[:, :, -1] = torch.sin(x[:, :, 2] * freqs[-1])
    
    assert torch.allclose(output, expected_output, rtol=1e-5)
    print("Embedding standard deviation =", output.std().item())
    
    
def test_sinusoidal_mlp_encoder():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define test parameters
    batch_size = 4
    set_size = 10
    D = 3
    d_model = 16
    max_freq = 8

    # Generate random input
    input_data = np.random.rand(batch_size, set_size, D)
    input_tensor = torch.tensor(input_data, dtype=torch.float)

    # Initialize encoder
    encoder = SinusoidalMLPEncoder(d_model, max_freq)

    # Test output shape
    output_tensor = encoder(input_tensor)
    expected_shape = (batch_size, set_size, d_model)
    assert output_tensor.shape == expected_shape, f"Output shape {output_tensor.shape} does not match expected shape {expected_shape}"

    # Test output range
    output_data = output_tensor.detach().numpy()
    print("Embedding standard deviation =", output_data.std().item())



def test_mlp_encoder():
    # Set random seed for reproducibility
    torch.manual_seed(0)

    # Define test parameters
    batch_size = 4
    set_size = 10
    D = 3
    d_model = 16

    # Generate random input
    input_data = np.random.rand(batch_size, set_size, D)
    input_tensor = torch.tensor(input_data, dtype=torch.float)

    # Initialize encoder
    encoder = MLPEncoder(D, d_model)

    # Test output shape
    output_tensor = encoder(input_tensor)
    expected_shape = (batch_size, set_size, d_model)
    assert output_tensor.shape == expected_shape, f"Output shape {output_tensor.shape} does not match expected shape {expected_shape}"

    # Test output range
    output_data = output_tensor.detach().numpy()
    print("Embedding standard deviation =", output_data.std().item())