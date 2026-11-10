import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedNet(nn.Module):
    def __init__(self):
        super().__init__()
        dilations = [1, 2 ,4, 8, 16, 1]

        layers = []
        in_ch = 3
        for d in dilations:
            layers.append(nn.Conv2d(in_ch, 64, 3, padding=d, dilation=d))
            in_ch = 64

        self.convs = nn.ModuleList(layers)
        self.out_conv = nn.Conv2d(64,64,1)


       
    def forward(self, x):
        for conv in self.convs:
            x = F.relu(conv(x))
        
        return self.out_conv(x)
