
import torch

def calculate_metrics(outputs, targets, smooth=1e-8):
    outputs = torch.sigmoid(outputs)
    preds = (outputs > 0.5).float()
    intersection = (preds * targets).sum()
    accuracy = (preds == targets).sum() / torch.numel(preds)
    dice = (2. * intersection) / (preds.sum() + targets.sum() + smooth)
    iou = intersection / (preds.sum() + targets.sum() - intersection + smooth)
    return accuracy.item(), dice.item(), iou.item()
