import torch.nn as nn

from .blocks import FEATURE_DIM, backbone_2d, head


class EarlyFusion(nn.Module):
    """All frames stacked as channels; the first 2D conv combines them."""

    def __init__(self, num_frames=10, num_classes=10, in_channels=3):
        super().__init__()
        self.backbone = backbone_2d(in_channels=in_channels * num_frames)
        self.head = head(FEATURE_DIM, num_classes)

    def forward(self, x):                          # x: [B, C, T, H, W]
        x = x.flatten(1, 2)                        # [B, C*T, H, W]
        return self.head(self.backbone(x))
