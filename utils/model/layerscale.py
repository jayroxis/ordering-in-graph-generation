import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """
    A normalization layer that scales the output of another layer
    by a learnable scalar value.

    This implementation of LayerScale is inspired by the CaiT model (Going 
    Deeper with Image Transformers).

    Args:
        d_model (int): The number of hidden units in the layer.
        init_values (float, optional): The initial value for the alpha parameter.
            Default is 1e-4.

    References:
        https://github.com/facebookresearch/deit/blob/ee8893c8063f6937fec7096e47ba324c206e22b9/cait_models.py#L129-L149
        Hugo Touvron, Matthieu Cord, Alexandre Sablayrolles, Gabriel Synnaeve, Hervé Jégou.
        "Going Deeper with Image Transformers." arXiv preprint arXiv:2103.17239 (2021).
    """

    def __init__(self, d_model, init_values=1.0, **kwargs):
        super(LayerScale, self).__init__()

        self.alpha = nn.Parameter(init_values * torch.ones(d_model))

    def forward(self, x):
        """
        Scales the input tensor by the learned scalar value.

        Args:
            x (torch.Tensor): The input tensor of shape (batch_size, sequence_length, d_model).

        Returns:
            torch.Tensor: The scaled output tensor of shape (batch_size, sequence_length, d_model).
        """
        return x * self.alpha.unsqueeze(0).unsqueeze(0)

