import torch
import torch.nn as nn
import numpy as np

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



@register_model
class GaussianEmbedding(nn.Module):
    def __init__(
        self, 
        num_tokens, 
        d_model, 
        trainable: bool = True,
        init_std: float = 0.02,
    ):
        """
        (Learnable) Positional Embedding initialized with Guassian Distribution.

        # initialize the positional embeddings:
        #    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        """
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens

        # initialize pos_emb to be guassian
        pos_emb = torch.randn(
            1, 
            self.num_tokens, 
            self.d_model
        ) * init_std

        # register pos_emb
        if trainable:
            self.register_parameter("pos_emb", nn.Parameter(pos_emb))
        else:
            self.register_buffer("pos_emb", pos_emb)

    def forward(self, x):
        return x + self.pos_emb
    

@register_model
class FourierEmbedding(nn.Module):
    def __init__(
        self, 
        num_tokens, 
        d_model, 
        trainable: bool = True,
        base_freq: int = 10000,
    ):
        """
        (Learnable) Positional Embedding initialized with Guassian Distribution.

        # initialize the positional embeddings:
        #    https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py
        """
        super().__init__()
        self.d_model = d_model
        self.num_tokens = num_tokens

        # initialize pos_emb to be fourier
        enc = FourierEncoder1D(d_model, base_freq=base_freq)
        pos_emb = enc(torch.randn(
            1, 
            self.num_tokens, 
            self.d_model,
        ))

        # register pos_emb
        if trainable:
            self.register_parameter("pos_emb", nn.Parameter(pos_emb))
        else:
            self.register_buffer("pos_emb", pos_emb)

    def forward(self, x):
        return x + self.pos_emb
    



#   ============    Fourier Embeddings   ============ 


def get_emb_from_value(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined.
    This sometimes is referred to as "juxtaposition" positional embedding.
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)


@register_model
class FourierEncoder1D(nn.Module):
    def __init__(
        self, 
        emb_dim, 
        base_freq: int = 10000, 
        trainable: bool = False
    ):
        """
        :param emb_dim: The last dimension of the tensor you want to apply pos emb to.
        """
        super(FourierEncoder1D, self).__init__()
        self.org_emb_dim = emb_dim
        emb_dim = int(np.ceil(emb_dim / 2) * 2)
        self.emb_dim = emb_dim
        inv_freq = 1.0 / (base_freq ** (torch.arange(0, emb_dim, 2).float() / emb_dim))
        if trainable:
            self.register_parameter("inv_freq", nn.Parameter(inv_freq))
        else:
            self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, length, emb_dim)
        :return: Positional Encoding Matrix of size (batch_size, length, emb_dim)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, length, emb_dim = tensor.shape
        pos_x = torch.arange(length, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb_from_value(sin_inp_x)
        emb = torch.zeros((length, self.emb_dim), device=tensor.device).type(tensor.type())
        emb[:, : self.emb_dim] = emb_x

        self.cached_penc = emb[None, :, :emb_dim].repeat(batch_size, 1, 1)
        return self.cached_penc



@register_model
class FourierEncoderPermute1D(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Accepts (batchsize, emb_dim, length) instead of (batchsize, length, emb_dim)
        """
        super(FourierEncoderPermute1D, self).__init__()
        self.penc = FourierEncoder1D(*args, **kwargs)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 2, 1)

    @property
    def org_emb_dim(self):
        return self.penc.org_emb_dim


@register_model
class FourierEncoder2D(nn.Module):
    def __init__(
        self, 
        channels, 
        base_freq: int = 10000, 
        trainable: bool = False
    ):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(FourierEncoder2D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 4) * 2)
        self.channels = channels
        inv_freq = 1.0 / (base_freq ** (torch.arange(0, channels, 2).float() / channels))
        if trainable:
            self.register_parameter("inv_freq", nn.Parameter(inv_freq))
        else:
            self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 4d tensor of size (batch_size, x, y, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, ch)
        """
        if len(tensor.shape) != 4:
            raise RuntimeError("The input tensor has to be 4d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        emb_x = get_emb_from_value(sin_inp_x).unsqueeze(1)
        emb_y = get_emb_from_value(sin_inp_y)
        emb = torch.zeros((x, y, self.channels * 2), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, : self.channels] = emb_x
        emb[:, :, self.channels : 2 * self.channels] = emb_y

        self.cached_penc = emb[None, :, :, :orig_ch].repeat(tensor.shape[0], 1, 1, 1)
        return self.cached_penc



@register_model
class FourierEncoderPermute2D(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Accepts (batchsize, ch, x, y) instead of (batchsize, x, y, ch)
        """
        super(FourierEncoderPermute2D, self).__init__()
        self.penc = FourierEncoder2D(*args, **kwargs)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 3, 1, 2)

    @property
    def org_channels(self):
        return self.penc.org_channels



@register_model
class FourierEncoder3D(nn.Module):
    def __init__(
        self, 
        channels, 
        base_freq: int = 10000, 
        trainable: bool = False
    ):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(FourierEncoder3D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 6) * 2)
        if channels % 2:
            channels += 1
        self.channels = channels
        inv_freq = 1.0 / (base_freq ** (torch.arange(0, channels, 2).float() / channels))
        if trainable:
            self.register_parameter("inv_freq", nn.Parameter(inv_freq))
        else:
            self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor):
        """
        :param tensor: A 5d tensor of size (batch_size, x, y, z, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, y, z, ch)
        """
        if len(tensor.shape) != 5:
            raise RuntimeError("The input tensor has to be 5d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, y, z, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        pos_z = torch.arange(z, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq)
        sin_inp_z = torch.einsum("i,j->ij", pos_z, self.inv_freq)
        emb_x = get_emb_from_value(sin_inp_x).unsqueeze(1).unsqueeze(1)
        emb_y = get_emb_from_value(sin_inp_y).unsqueeze(1)
        emb_z = get_emb_from_value(sin_inp_z)
        emb = torch.zeros((x, y, z, self.channels * 3), device=tensor.device).type(
            tensor.type()
        )
        emb[:, :, :, : self.channels] = emb_x
        emb[:, :, :, self.channels : 2 * self.channels] = emb_y
        emb[:, :, :, 2 * self.channels :] = emb_z

        self.cached_penc = emb[None, :, :, :, :orig_ch].repeat(batch_size, 1, 1, 1, 1)
        return self.cached_penc



@register_model
class FourierEncoderPermute3D(nn.Module):
    def __init__(self, *args, **kwargs):
        """
        Accepts (batchsize, ch, x, y, z) instead of (batchsize, x, y, z, ch)
        """
        super(FourierEncoderPermute3D, self).__init__()
        self.penc = FourierEncoder3D(*args, **kwargs)

    def forward(self, tensor):
        tensor = tensor.permute(0, 2, 3, 4, 1)
        enc = self.penc(tensor)
        return enc.permute(0, 4, 1, 2, 3)

    @property
    def org_channels(self):
        return self.penc.org_channels

