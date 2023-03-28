



import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from timm.models.registry import register_model


def permutation_invariant_errors(x, y, p=2, root=True, pad_value=-1):
    """
    Compute the mean of the minimum distances between each row of x and larger-indexed rows of y, 
    using the p-norm to calculate distances.

    This is specifically for GraphGPT since the node pair generation does not have a natural order
    to follow in constrast to natural languages. This function is designed for causal generative
    training while we don't want to enforce strict ordering the the sequence generation.

    Args:
    - x (torch.Tensor): a 3D tensor of shape (B, L, N) representing M N-dimensional vectors.
    - y (torch.Tensor): a 3D tensor of shape (B, L, N) representing K N-dimensional vectors.
    - p (int, optional): The order of the p-norm to be used in distance calculation. Default is 2, 
      which corresponds to Euclidean distance.
    - root (Bool, optional): If True, return the normalized distance or return the unnormalized 
      distance if False. Default is True.

    Returns:
    - errors.mean() (torch.Tensor): the minimum distances of shape (B, L) between each row of x and 
      larger-indexed rows of y.
    """

    assert x.shape == y.shape, "x and y must have the same shape"
    B, L, N = x.shape
    pad_mask = (y == pad_value)

    # a large value for masking positions
    large_value = 100   # if use torch.inf, the loss will be nan

    # Unpad the stop token
    unpadded = y.clone()
    unpadded[pad_mask] = large_value

    # Compute pairwise p-norm distances between rows of x and y
    dists = torch.cdist(x, unpadded, p=p)
    if not root:
        dists = dists ** p
    
    # Create a mask to ignore distances between rows of y with smaller index
    mask = torch.triu(torch.ones_like(dists), diagonal=0)
    mask[mask == 0] = large_value
    
    # Compute the errors for each row in x as the minimum distance with larger-indexed rows of y
    errors, _ = (dists * mask).min(dim=-1)

    # Restore the stop token prediction error.
    orig_err = torch.linalg.norm(x - y, ord=p, dim=-1)
    errors[pad_mask.all(-1)] = orig_err[pad_mask.all(-1)]

    return errors / N


@register_model
class UndirectedGraphLoss(nn.Module):
    """
    L1 + L2 Loss for node pairs on an undirected graph.
    """
    def __init__(self):
        super(UndirectedGraphLoss, self).__init__()
        
    def forward(self, pred, target):
        """
        Compute the loss between predicted and target node pairs.
        
        Args:
        - pred (torch.Tensor): predicted node pairs.
        - target (torch.Tensor): target node pairs.
        
        Returns:
        - final_loss (torch.Tensor): computed loss between predicted and target.
        """
        
        # Get the last dimension of the target tensor
        target_dim = target.shape[-1]

        # Compute the half of the target dimension
        half_target_dim = int(target_dim // 2)
        
        # Flip the start and end node of the target graph
        reversed_target = torch.cat([
            target[..., half_target_dim:], 
            target[..., :half_target_dim]
        ], dim=-1)

        # Compute MSE loss on original target and reversed target
        mse_loss_1 = ((pred - target) ** 2).mean(-1)
        mse_loss_2 = ((pred - reversed_target) ** 2).mean(-1)
        mse_loss = torch.minimum(mse_loss_1, mse_loss_2)
        
        # Compute L1 loss on original target and reversed target
        l1_loss_1 = torch.abs(pred - target).mean(-1)
        l1_loss_2 = torch.abs(pred - reversed_target).mean(-1)
        l1_loss = torch.minimum(l1_loss_1, l1_loss_2)
        
        # Compute final loss as the sum of MSE and L1 losses
        final_loss = mse_loss + l1_loss
        return final_loss.mean()



class UnorderedUndirectedGraphLoss(nn.Module):
    """
    L1 + L2 Loss for node pairs on an undirected graph. The errors
    are calculated using permutation invariant error that does not 
    enforce a strict ordering in comparing prediction vs GT.
    """
    def __init__(self):
        super(UnorderedUndirectedGraphLoss, self).__init__()
        
    def forward(self, pred, target):
        target_dim = target.shape[-1]
        half_target_dim = int(target_dim // 2)
        
        # target that flip the start and end node
        reversed_target = torch.cat([
            target[..., half_target_dim:], 
            target[..., :half_target_dim]
        ], dim=-1)

        # Compute L2 loss on original target and reversed target
        loss_func = permutation_invariant_errors
        mse_loss_1 = loss_func(x=pred, y=target, p=2)
        mse_loss_2 = loss_func(x=pred, y=reversed_target, p=2)
        mse_loss = torch.minimum(mse_loss_1, mse_loss_2)
        
        # Compute L1 loss on original target and reversed target
        l1_loss_1 = loss_func(x=pred, y=target, p=1)
        l1_loss_2 = loss_func(x=pred, y=reversed_target, p=1)
        l1_loss = torch.minimum(l1_loss_1, l1_loss_2)
        
        # Compute final loss as the sum of MSE and L1 losses
        final_loss = mse_loss + l1_loss
        if torch.isnan(final_loss).any() or torch.isinf(final_loss).any():
            import pdb; pdb.set_trace()
        return final_loss.mean()
    



@register_model  
class TorchMetricsMulticlass(nn.Module):
    def __init__(self, metric: str, dim: int = -1, **kwargs):
        super().__init__()
        self.dim = dim
        average = kwargs.get("average", "weighted")
        self.metric = eval(metric)(
            average=average,
            **kwargs
        )
    
    def _flatten_along_dim(self, x):
        """
        Flatten the tensor except the given dimension.
        """
        x = x.transpose(-1, self.dim)
        x = x.flatten(end_dim=-2)
        return x
    
    def forward(self, pred, true):
        pred = self._flatten_along_dim(pred)
        true = self._flatten_along_dim(true)
        true = true / (
            true.abs().sum(self.dim).unsqueeze(self.dim) + 1e-9
        )
        pred = pred / (
            pred.abs().sum(self.dim).unsqueeze(self.dim) + 1e-9
        )
        true = torch.argmax(true, dim=self.dim)
        return self.metric(preds=pred, target=true)



@register_model
class CrossEntropyWithNormalize(nn.CrossEntropyLoss):
    """
    Cross-Entropy Loss that Normalize the Target Labels.
    """
    def forward(self, input, target):
        target = target / (target.abs().sum(-1).unsqueeze(-1) + 1e-9)
        return super().forward(input=input, target=target)


@register_model
class DimensionwiseHybridLoss(nn.Module):
    def __init__(self, config, **kwargs):
        super(DimensionwiseHybridLoss, self).__init__()
        self.loss_functions = []
        for d in config:
            loss_fn = eval(d["class"])(**d.get("params", {}))
            start_idx, end_idx = d["index"].split(":")
            start_idx = int(start_idx) if start_idx != ""  else None
            end_idx = int(end_idx) if end_idx != ""  else None
            weight = d.get("weight", 1.0)
            self.loss_functions.append((loss_fn, weight, start_idx, end_idx))

    def forward(self, pred, target):
        loss = 0.0
        for loss_fn, weight, start_idx, end_idx in self.loss_functions:
            # Apply the corresponding loss function for the specified dimensions
            loss = loss + weight * loss_fn(
                pred[..., start_idx:end_idx], 
                target[..., start_idx:end_idx]
            )
        return loss

    def __repr__(self):
        table = [(i, str(loss_fn), weight, start_idx, end_idx)
                 for i, (loss_fn, weight, start_idx, end_idx) in enumerate(self.loss_functions)]
        headers = ["Index", "Loss Function", "Weight", "Start Index", "End Index"]
        repr_str = f"{self.__class__.__name__}():\n"
        # Build the table string manually
        repr_str += f"{headers[0]:^6} {headers[1]:^35} {headers[2]:^7} {headers[3]:^12} {headers[4]:^10}\n"
        for row in table:
            repr_str += f"{row[0]:^6} {row[1]:^35} {row[2]:^7} {row[3]:^12} {row[4]:^10}\n"
        return repr_str


@register_model
class PSGRelationalLoss(nn.Module):
    def __init__(
        self,
        object_classes: int = 133,
        predicate_classes: int = 56,
        loss_func: dict = {"class": "nn.CrossEntropyLoss"},
    ):
        """
        This module is for target relational labels are not one-hot
        encoded. 
        For one-hot encoded labels. Use the `DimensionwiseHybridLoss`.
        """
        super(PSGRelationalLoss, self).__init__()
        loss_class = eval(loss_func["class"])
        loss_params = loss_func.get("params", {})
        self.loss_func = loss_class(**loss_params)
        self.obj_cls = object_classes
        self.pd_cls = predicate_classes

    def forward(self, pred, target):
        # Convert target to one-hot
        gt_triplelet = [
            F.one_hot(target[..., 0], num_classes=self.obj_cls).float(),
            F.one_hot(target[..., 1], num_classes=self.pd_cls).float(),
            F.one_hot(target[..., 2], num_classes=self.obj_cls).float(),
        ]
        pred_triplelet = [
            pred[..., :self.obj_cls],
            pred[..., self.obj_cls:-self.obj_cls],
            pred[..., -self.obj_cls:],
        ]
        loss = 0
        for i in range(3):
            print(pred_triplelet[i], gt_triplelet[i])
            loss = loss + self.loss_func(pred_triplelet[i], gt_triplelet[i])
        return loss