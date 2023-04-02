



import torch
import torch.nn as nn
import torch.nn.functional as F

import torchmetrics
from timm.models.registry import register_model
from scipy.optimize import linear_sum_assignment


# ====================== Distance Metrics ========================

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


def pairwise_undirected_graph_distance(x, y):
    """
    Computes pairwise undirected graph distance between two sets of vectors x and y.
    
    Args:
        x: Tensor of shape (B, L, D).
        y: Tensor of shape (B, L, D).
        
    Returns:
        Tensor of shape (B, L, L) with pairwise undirected graph distances.
    """
    assert x.ndim == 3, f"x must be a 3-D tensor, got {x.shape} instead."
    B, L, D = x.shape
    assert D % 2 == 0, f"D must be an even number, got {D} instead."
    half_D = int(D // 2)

    x = x.unsqueeze(2)
    y = y.unsqueeze(1)

    y_reversed = torch.cat([y[..., half_D:], y[..., :half_D]], dim=-1)

    mse_loss_1 = ((x - y) ** 2).mean(-1)
    mse_loss_2 = ((x - y_reversed) ** 2).mean(-1)
    mse_loss = torch.minimum(mse_loss_1, mse_loss_2)

    l1_loss_1 = torch.abs(x - y).mean(-1)
    l1_loss_2 = torch.abs(x - y_reversed).mean(-1)
    l1_loss = torch.minimum(l1_loss_1, l1_loss_2)

    distance = mse_loss + l1_loss
    return distance



# ====================== PyTorch Loss Modules ========================


@register_model
class CustomCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    A custom version of the `nn.CrossEntropyLoss` loss function that allows
    the user to specify the input dimension.
    
    This is useful when the input tensor has more than two dimensions and
    the user wants to apply the loss function along a different dimension
    than the default second dimension. By default, `nn.CrossEntropyLoss`
    applies the loss function to the second dimension of the input tensor.
    
    Parameters:
        dim (int, optional): The input dimension to apply the loss function along.
                             Default is -1 (last dimension).
        *args: Variable length argument list to pass to the parent constructor.
        **kwargs: Arbitrary keyword arguments to pass to the parent constructor.
    """
    
    def __init__(self, dim=-1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim = dim
    
    def forward(self, input, target, *args, **kwargs):
        """
        Applies the custom `nn.CrossEntropyLoss` loss function to the input tensor
        along the specified dimension.
        
        Parameters:
            input (Tensor): Input tensor of shape (N, C, ...) where N is the batch size,
                            C is the number of classes, and ... are additional dimensions.
            target (Tensor): Target tensor of shape (N, ...) where N is the batch size
                             and ... are additional dimensions.
            *args: Variable length argument list to pass to the parent forward() method.
            **kwargs: Arbitrary keyword arguments to pass to the parent forward() method.
        
        Returns:
            Tensor: The calculated loss tensor of shape (N, ...).
        """
        # Swap the last and target dimension
        input = input.transpose(1, self.dim)
        target = target.transpose(1, self.dim)
        
        # Apply the original CrossEntropyLoss function
        loss = super().forward(input, target, *args, **kwargs)
        
        return loss



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



@register_model
class HungarianLoss(nn.Module):
    """
    Hungarian Loss for set similarity. Sparse form: O(n^3)
    """
    def __init__(self, pairwise_distances=None, reduction="mean"):
        super(HungarianLoss, self).__init__()
        self.reduction = reduction
        if pairwise_distances is None:
            self.pairwise_distances = self._pairwise_distances
        elif type(pairwise_distances) == str:
            self.pairwise_distances = eval(pairwise_distances)
        elif type(pairwise_distances) == dict:
            pd = pairwise_distances
            self.pairwise_distances = eval(pd["class"])(**(pd.get("params", {})))
        else:
            self.pairwise_distances = pairwise_distances

    def _pairwise_distances(self, x, y):
        """
        Computes pairwise distances between two sets of vectors x and y.
        This will be used as default if no external metric is provided.
        Args:
            x: Tensor of shape (B, L, D).
            y: Tensor of shape (B, L, D).
        Returns:
            Tensor of shape (B, L, L) with pairwise distances.
        """
        x = x.unsqueeze(2)
        y = y.unsqueeze(1)
        return torch.norm(x - y, dim=-1)

    def forward(self, pred, target):
        """
        Compute the Hungarian loss between the predicted set and target set.
        Args:
            pred: Tensor of shape (B, L, D), predicted set of vectors.
            target: Tensor of shape (B, L, D), target set of vectors.
        Returns:
            Scalar loss value.
        """
        assert pred.shape == target.shape
        B, L, _ = pred.shape
        total_loss = 0.0

        pairwise_dist = self.pairwise_distances(pred, target)

        for b in range(B):
            cost_matrix = pairwise_dist[b].detach().cpu().numpy()
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
            if self.reduction == "sum":
                total_loss += pairwise_dist[b, row_ind, col_ind].sum()
            else:
                total_loss += pairwise_dist[b, row_ind, col_ind].mean()
        return total_loss / B



# @register_model
# class SinkhornLoss(nn.Module):
#     """
#     Sinkhorn-Knopp Loss for set similarity. Dense form: O(n^2 log n)
#     [WARNING]: This may result in inaccurate prediction, use with caution!
#     If you do not know when to use this loss, use Hungarian Loss instead.
#     """
#     def __init__(self, pairwise_distances=None, epsilon=0.1, max_iter=50, reduction="mean"):
#         super(SinkhornLoss, self).__init__()
#         self.epsilon = epsilon
#         self.max_iter = max_iter
#         self.reduction = reduction
#         if pairwise_distances is None:
#             self.pairwise_distances = self._pairwise_distances
#         elif type(pairwise_distances) == str:
#             self.pairwise_distances = eval(pairwise_distances)
#         elif type(pairwise_distances) == dict:
#             pd = pairwise_distances
#             self.pairwise_distances = eval(pd["class"])(**(pd.get("params", {})))
#         else:
#             self.pairwise_distances = pairwise_distances

#     def _pairwise_distances(self, x, y):
#         """
#         Computes pairwise distances between two sets of vectors x and y.
#         This will be used as default if no external metric is provided.
#         Args:
#             x: Tensor of shape (B, L, D).
#             y: Tensor of shape (B, L, D).
#         Returns:
#             Tensor of shape (B, L, L) with pairwise distances.
#         """
#         x = x.unsqueeze(2)
#         y = y.unsqueeze(1)
#         return torch.norm(x - y, dim=3)

#     # def sinkhorn_iter(self, cost_matrix):
#     #     B, L, _ = cost_matrix.shape
#     #     K = torch.exp(-self.epsilon * cost_matrix)
#     #     u = torch.ones(B, L, 1, device=K.device) / L
#     #     for _ in range(self.max_iter):
#     #         v = 1.0 / (torch.matmul(K.transpose(1, 2), u) + 1e-8)
#     #         u = 1.0 / (torch.matmul(K, v) + 1e-8)
#     #     P = u * K * v.transpose(1, 2)
#     #     return P

#     def sinkhorn_iter(self, cost_matrix):
#         B, L, _ = cost_matrix.shape
#         K = torch.exp(-self.epsilon * cost_matrix)
#         u = torch.ones(B, L, 1, device=K.device) / L
#         for _ in range(self.max_iter):
#             v = 1.0 / (torch.matmul(K.transpose(1, 2), u) + 1e-8)
#             u = 1.0 / (torch.matmul(K, v) + 1e-8)
#             u = u / u.sum(dim=1, keepdim=True)  # Row normalization
#             v = v / v.sum(dim=2, keepdim=True)  # Column normalization
#         P = u * K * v.transpose(1, 2)
#         return P

#     # def forward(self, pred, target):
#     #     assert pred.shape == target.shape
#     #     B, L, _ = pred.shape
#     #     total_loss = 0.0

#     #     pairwise_dist = self.pairwise_distances(pred, target)
#     #     pairwise_dist = pairwise_dist - pairwise_dist.min(dim=1, keepdim=True)[0]
#     #     pairwise_dist = pairwise_dist - pairwise_dist.min(dim=2, keepdim=True)[0]
#     #     max_abs = pairwise_dist.abs().max()
#     #     pairwise_dist = pairwise_dist / max_abs + 1e-8
#     #     P = self.sinkhorn_iter(pairwise_dist)

#     #     for b in range(B):
#     #         if self.reduction == "sum":
#     #             total_loss += torch.sum(P[b] * pairwise_dist[b])
#     #         else:
#     #             total_loss += torch.mean(P[b] * pairwise_dist[b])
#     #     return total_loss / B

#     def forward(self, pred, target):
#         assert pred.shape == target.shape
#         B, L, _ = pred.shape
#         total_loss = 0.0

#         pairwise_dist = self.pairwise_distances(pred, target)
#         P = self.sinkhorn_iter(pairwise_dist)

#         for b in range(B):
#             if self.reduction == "sum":
#                 total_loss += torch.sum(P[b] * pairwise_dist[b])
#             else:
#                 total_loss += torch.mean(P[b] * pairwise_dist[b])

#         return (total_loss + 1e-8) / B



@register_model
class UnorderedUndirectedGraphLoss(nn.Module):
    """
    Similar to Hungarian loss but used for Auto-Regressive model.
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
    
    @torch.no_grad()
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


