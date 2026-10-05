import torch.nn as nn

from .blocks import FEATURE_DIM, backbone_2d, head


class PerFrame(nn.Module):
    """Classifies each frame independently. Returns per-frame logits [B, T, C];
    the training script averages the softmax over frames to get a video prediction."""

    def __init__(self, num_frames=10, num_classes=10):
        super().__init__()
        self.backbone = backbone_2d(in_channels=3)
        self.head = head(FEATURE_DIM, num_classes)

    def forward(self, x):                          # x: [B, 3, T, H, W]
        B, C, T, H, W = x.shape
        x = x.transpose(1, 2).reshape(B * T, C, H, W)
        logits = self.head(self.backbone(x))       # [B*T, num_classes]
        return logits.view(B, T, -1)
