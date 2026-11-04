import torch
import torch.nn.functional as F
import torch.nn as nn


class QuantileLoss(nn.Module):

    def __init__(self, q=0.5):

        super().__init__()

        self.q = q

    def forward(self, pred, target):

        error = target - pred

        return torch.mean(torch.maximum(self.q * error, (self.q - 1) * error))
