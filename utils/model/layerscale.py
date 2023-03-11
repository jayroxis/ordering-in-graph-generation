import torch
import torch.nn as nn


class LayerScale(nn.Module):
    """
    A scaling layer that scales the output of another 
    layer by a learned scalar value.
    """

    def __init__(self, d_model):
        """
        Initializes the LayerScale.

        Args:
        - d_model (int): The number of hidden units in the layer.
        """
        super(LayerScale, self).__init__()

        self.alpha = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        """
        Scales the input tensor by the learned scalar value.

        Args:
        - x (torch.Tensor): The input tensor of shape 
            (batch_size, sequence_length, d_model).

        Returns:
        - torch.Tensor: The scaled output tensor of shape 
          (batch_size, sequence_length, d_model).
        """
        return x * self.alpha.unsqueeze(0).unsqueeze(0)