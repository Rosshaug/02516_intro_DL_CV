import torch
import torch.nn as nn

class BCELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        #loss = torch.mean(y_pred - y_true*y_pred + torch.log(1 + torch.exp(-y_pred)))
        loss = torch.mean(torch.clamp(y_pred, min=0) - y_pred * y_true + torch.log1p(torch.exp(-torch.abs(y_pred))))
        return loss

class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        p = torch.sigmoid(y_pred)
        top = torch.mean(2*p * y_true + 1)
        bot = torch.mean(y_true + p) + 1
        loss = 1 - top / bot
        return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        gamma = 2

        p = torch.sigmoid(y_pred)

        inner = inner = (1-p)**gamma * y_true * nn.functional.logsigmoid(y_pred) + p**gamma * (1-y_true) * nn.functional.logsigmoid(-y_pred)
        loss = -torch.mean(inner)
        return loss

class BCELoss_TotalVariation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(torch.clamp(y_pred, min=0) - y_pred * y_true + torch.log1p(torch.exp(-torch.abs(y_pred))))
        p = torch.sigmoid(y_pred)
        regularization = (p[..., 1:, :] - p[..., :-1, :]).abs().mean() + (p[..., :, 1:] - p[..., :, :-1]).abs().mean()
        return loss + 0.1*regularization

