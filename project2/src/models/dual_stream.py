import torch
import torch.nn as nn

from .blocks import FEATURE_DIM, backbone_2d, head

class DualStream(nn.Module):
    """
    Two-Stream Network:
    - Spatial Stream: Analyzes RGB frames independently, then aggregates features (Late Fusion).
    - Temporal Stream: Analyzes stacked optical flow fields using a single 2D CNN (Early Fusion).
    - Fusion: Averages the logits of both streams to produce the final prediction.
    """

    def __init__(self, num_frames=10, num_classes=10):
        super().__init__()
        
        # --- Spatial Stream ---
        self.spatial_backbone = backbone_2d(in_channels=3)
        self.spatial_head = head(num_frames * FEATURE_DIM, num_classes, hidden=256)
        
        # --- Temporal Stream ---
        num_flow_channels = 2 * (num_frames - 1)
        self.temporal_backbone = backbone_2d(in_channels=num_flow_channels)
        self.temporal_head = head(FEATURE_DIM, num_classes)
        
        # --- Weight Initialization Trick ---
        spatial_conv_weight = self.spatial_backbone[0][0].weight.data  # shape: [32, 3, 3, 3]
        temporal_conv_weight = spatial_conv_weight.mean(dim=1, keepdim=True)  # shape: [32, 1, 3, 3]
        temporal_conv_weight = temporal_conv_weight.repeat(1, num_flow_channels, 1, 1) # shape: [32, 18, 3, 3]
        self.temporal_backbone[0][0].weight.data = temporal_conv_weight

    def forward(self, rgb, flow):
        # rgb shape: [B, 3, T, H, W]
        # flow shape: [B, 18, H, W]
        
        # 1. Spatial Stream Processing
        B, C, T, H, W = rgb.shape
        rgb = rgb.transpose(1, 2).reshape(B * T, C, H, W)
        spatial_features = self.spatial_backbone(rgb).view(B, -1) # [B, T * FEATURE_DIM]
        spatial_logits = self.spatial_head(spatial_features)      # [B, num_classes]
        
        # 2. Temporal Stream Processing
        temporal_features = self.temporal_backbone(flow)          # [B, FEATURE_DIM]
        temporal_logits = self.temporal_head(temporal_features)   # [B, num_classes]
        
        # 3. Late Score Fusion
        fused_logits = (spatial_logits + temporal_logits) / 2.0
        
        # We return all three so your training loop can evaluate individual streams vs fused performance
        return fused_logits, spatial_logits, temporal_logits