import torch
import torch.nn.functional as torch_functional
import torch.nn as nn

import segmentation_models_pytorch as smp

def dice_loss_from_logits_multiclass(logits, targets, eps=1e-7):
    dice = smp.losses.DiceLoss(mode='multiclass', from_logits=True)(logits,targets)
    return dice

def combined_multiclass_loss(logits, targets, dice_weight=0.5, ce_weight=0.5):
    dice = dice_loss_from_logits_multiclass(logits, targets)
    ce = torch_functional.cross_entropy(logits, targets)

    return dice_weight * dice + ce_weight * ce

class DiceCELossModule(nn.Module):
    def __init__(self, dice_weight: float = 0.5, ce_weight: float = 0.5, smooth: float = 1e-6):
        super().__init__()

        self.dice_weight = dice_weight
        self.bce_weight = ce_weight
        self.smooth = smooth
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, targets):
        targets = targets.squeeze(1).long()
        ce_loss = self.ce(logits, targets)

        dice_loss = dice_loss_from_logits_multiclass(logits,targets,self.smooth)

        total_loss = self.dice_weight * dice_loss + self.bce_weight * ce_loss

        return total_loss