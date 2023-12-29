
import torch
import torch.nn as nn
import torch.nn.functional as F

class LungLoss(nn.Module):
    """BCE, Jaccard_loss and Dice_loss"""
    def __init__(self, weight=None, size_average=True):
        super(LungLoss, self).__init__()

    def forward(self, outputs, targets, smooth=1e-8):
        outputs = torch.sigmoid(outputs)
        outputs = outputs.view(-1)
        targets = targets.view(-1)
        intersection = (outputs * targets).sum()
        union = outputs.sum() + targets.sum() - intersection
        jaccard_loss = 1 - (intersection / (union + smooth))
        dice_loss = 1 - (2. * intersection) / (outputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(outputs, targets, reduction='mean')
        loss = BCE + jaccard_loss + dice_loss
        return loss
