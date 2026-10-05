import torch.nn as nn

from .blocks import FEATURE_DIM, backbone_3d, head


class CNN3D(nn.Module):
    """3D convolutions, so time is combined gradually through the network."""

    def __init__(self, num_frames=10, num_classes=10):
        super().__init__()
        self.backbone = backbone_3d(in_channels=3)
        self.head = head(FEATURE_DIM, num_classes)

    def forward(self, x):                          # x: [B, 3, T, H, W]
        return self.head(self.backbone(x))
