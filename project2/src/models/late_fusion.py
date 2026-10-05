import torch.nn as nn

from .blocks import FEATURE_DIM, backbone_2d, head


class LateFusion(nn.Module):
    """Shared 2D backbone on each frame, frame features concatenated and fed to an MLP."""

    def __init__(self, num_frames=10, num_classes=10):
        super().__init__()
        self.backbone = backbone_2d(in_channels=3)
        self.head = head(num_frames * FEATURE_DIM, num_classes, hidden=256)

    def forward(self, x):                          # x: [B, 3, T, H, W]
        B, C, T, H, W = x.shape
        x = x.transpose(1, 2).reshape(B * T, C, H, W)
        features = self.backbone(x).view(B, -1)    # [B, T * FEATURE_DIM]
        return self.head(features)
