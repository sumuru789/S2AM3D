import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        pos_weight = target.sum() / target.numel()
        beta = (1 - pos_weight) / (pos_weight + self.epsilon)
        bce_loss = F.binary_cross_entropy(
            pred, target, weight=target * beta + (1 - target)
        )
        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1) + self.epsilon
        dice_loss = (1 - (2 * intersection + self.epsilon) / union).mean()
        total_loss = 0.7 * bce_loss + 0.3 * dice_loss
        if total_loss.isnan() or total_loss.isinf():
            return 0.7 * bce_loss
        return total_loss
