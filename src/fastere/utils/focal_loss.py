"""Focal loss for class-imbalanced relation extraction."""

import torch
import torch.nn as nn
from torch.nn import functional as F


class focal_loss(nn.Module):
    def __init__(self, alpha=None, gamma=2, num_classes=3, size_average=True):
        super().__init__()
        if alpha is None:
            self.alpha = torch.ones(num_classes)
        elif isinstance(alpha, list):
            assert len(alpha) == num_classes
            self.alpha = torch.Tensor(alpha)
        else:
            assert alpha < 1
            self.alpha = torch.Tensor([alpha] + [1 - alpha] * (num_classes - 1))
            self.alpha = self.alpha / self.alpha.sum() * num_classes

        self.gamma = gamma
        self.num_classes = num_classes
        self.size_average = size_average

    def forward(self, preds, labels):
        preds = preds.view(-1, preds.size(-1))
        alpha = self.alpha.to(preds.device)
        preds_logsoft = F.log_softmax(preds, dim=1)
        preds_softmax = torch.exp(preds_logsoft)

        preds_softmax = preds_softmax.gather(1, labels.view(-1, 1))
        preds_logsoft = preds_logsoft.gather(1, labels.view(-1, 1))
        alpha = alpha.gather(0, labels.view(-1))
        loss = -alpha * (1 - preds_softmax) ** self.gamma * preds_logsoft

        return loss.mean() if self.size_average else loss.sum()
