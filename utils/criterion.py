
import torch
import torch.nn as nn


class UndirectedGraphLoss(nn.Module):
    def __init__(self):
        super(UndirectedGraphLoss, self).__init__()
        
    def forward(self, pred, target):
        target_dim = target.shape[-1]
        half_target_dim = int(target_dim // 2)
        
        # Compute MSE loss on original target and reversed target
        mse_loss_1 = ((pred - target) ** 2).mean(-1)
        reversed_target = torch.cat([
            target[..., half_target_dim:], 
            target[..., :half_target_dim]
        ], dim=-1)
        mse_loss_2 = ((pred - reversed_target) ** 2).mean(-1)
        mse_loss = torch.minimum(mse_loss_1, mse_loss_2)
        
        # Compute L1 loss on original target and reversed target
        l1_loss_1 = torch.abs(pred - target).mean(-1)
        l1_loss_2 = torch.abs(pred - reversed_target).mean(-1)
        l1_loss = torch.minimum(l1_loss_1, l1_loss_2)
        
        # Compute final loss as the sum of MSE and L1 losses
        final_loss = mse_loss + l1_loss
        return final_loss.mean()
