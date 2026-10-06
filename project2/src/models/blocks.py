"""Shared building blocks. Every model is assembled from these, so the only
difference between models is where frames are combined."""
import torch.nn as nn

CHANNELS = [32, 64, 128, 256]
FEATURE_DIM = CHANNELS[-1]


def conv_block_2d(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
    )


def conv_block_3d(in_channels, out_channels, pool):
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
        nn.MaxPool3d(pool),
    )


def backbone_2d(in_channels=3):
    """[N, in_channels, H, W] -> [N, FEATURE_DIM]"""
    layers = []
    for out_channels in CHANNELS:
        layers.append(conv_block_2d(in_channels, out_channels))
        in_channels = out_channels
    return nn.Sequential(*layers, nn.AdaptiveAvgPool2d(1), nn.Flatten())


def backbone_3d(in_channels=3):
    """[N, in_channels, T, H, W] -> [N, FEATURE_DIM]"""
    # First pool keeps time intact (as in C3D); T=10 -> 10 -> 5 -> 2 -> 1.
    pools = [(1, 2, 2), (2, 2, 2), (2, 2, 2), (2, 2, 2)]
    layers = []
    for out_channels, pool in zip(CHANNELS, pools):
        layers.append(conv_block_3d(in_channels, out_channels, pool))
        in_channels = out_channels
    return nn.Sequential(*layers, nn.AdaptiveAvgPool3d(1), nn.Flatten())


def head(in_features, num_classes, hidden=None):
    """Linear classifier, or a one-hidden-layer MLP if `hidden` is given.
    Both variants have exactly one dropout, right before the final layer."""
    if hidden is None:
        return nn.Sequential(nn.Dropout(0.5), nn.Linear(in_features, num_classes))
    return nn.Sequential(
        nn.Linear(in_features, hidden),
        nn.ReLU(inplace=True),
        nn.Dropout(0.5),
        nn.Linear(hidden, num_classes),
    )
