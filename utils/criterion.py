
import torch
import torch.nn as nn



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



class LastTokenMatchLoss(nn.Module):
    def __init__(self, pad_value=-1, fill_value=1e9):
        super(LastTokenMatchLoss, self).__init__()
        self.pad_value = pad_value
        self.fill_value = fill_value

    def forward(self, pred, target):
        pred = pred[:, -1:]
        unmaksed_idx = (target != self.pad_value).all(-1)
        elu_dist = torch.cdist(pred, target, p=2.0).squeeze(1)
        abs_dist = torch.cdist(pred, target, p=1.0).squeeze(1)
        dist = (elu_dist + abs_dist)
        dist[~unmaksed_idx] = self.fill_value
        min_dist = dist.min(-1).values
        loss = min_dist.mean()
        return loss
