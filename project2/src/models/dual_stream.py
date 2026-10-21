import torch
import torch.nn as nn

from .blocks import FEATURE_DIM, backbone_2d, head


class DualStream(nn.Module):
    """
    Two-Stream Network:
    - Spatial Stream: Evaluates RGB frames independently and averages frame logits.
    - Temporal Stream: Analyzes stacked optical flow fields [B, 18, H, W] via early fusion.
    - Fusion: Averages the predictions of both streams.
    """

    def __init__(self, num_frames=10, num_classes=10):
        super().__init__()

        # --- Spatial Stream (Per-Frame 2D CNN) ---
        self.spatial_backbone = backbone_2d(in_channels=3)
        self.spatial_head = head(FEATURE_DIM, num_classes)

        # --- Temporal Stream (Early-Fusion 2D CNN on 18 flow channels) ---
        num_flow_channels = 2 * (num_frames - 1)
        self.temporal_backbone = backbone_2d(in_channels=num_flow_channels)
        self.temporal_head = head(FEATURE_DIM, num_classes)

    def forward(self, rgb, flow):
        # rgb shape: [B, 3, T, H, W]
        # flow shape: [B, 18, H, W]

        # 1. Spatial Stream Processing (Per-frame -> average across T)
        B, C, T, H, W = rgb.shape
        rgb_flat = rgb.transpose(1, 2).reshape(B * T, C, H, W)
        spatial_features = self.spatial_backbone(rgb_flat)             # [B * T, FEATURE_DIM]
        spatial_frame_logits = self.spatial_head(spatial_features)     # [B * T, num_classes]
        spatial_logits = spatial_frame_logits.view(B, T, -1).mean(1)   # [B, num_classes]

        # 2. Temporal Stream Processing
        temporal_features = self.temporal_backbone(flow)               # [B, FEATURE_DIM]
        temporal_logits = self.temporal_head(temporal_features)        # [B, num_classes]

        # 3. Late Score Fusion
        fused_logits = 0.5 * spatial_logits + 0.5 * temporal_logits

        return fused_logits, spatial_logits, temporal_logits