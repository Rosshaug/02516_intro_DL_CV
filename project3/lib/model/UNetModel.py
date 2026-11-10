import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.pool0 = nn.MaxPool2d(2, 2)  # H -> H/2
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)  # H/2 -> H/4
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)  # H/4 -> H/8
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.pool3 = nn.MaxPool2d(2, 2)  # H/8 -> H/16

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2)  # H/16 -> H/8
        self.dec_conv0 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample1 = nn.Upsample(scale_factor=2)  # H/8 -> H/4
        self.dec_conv1 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample2 = nn.Upsample(scale_factor=2)  # H/4 -> H/2
        self.dec_conv2 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample3 = nn.Upsample(scale_factor=2)  # H/2 -> H
        self.dec_conv3 = nn.Conv2d(64*2, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        c0 = F.relu(self.enc_conv0(x))
        e0 = self.pool0(c0)
        c1 = F.relu(self.enc_conv1(e0))
        e1 = self.pool1(c1)
        c2 = F.relu(self.enc_conv2(e1))
        e2 = self.pool2(c2)
        c3 = F.relu(self.enc_conv3(e2))
        e3 = self.pool3(c3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        i0 = torch.cat([c3,self.upsample0(b)], dim=1)
        d0 = F.relu(self.dec_conv0(i0))
        i1 = torch.cat([c2,self.upsample1(d0)], dim=1)
        d1 = F.relu(self.dec_conv1(i1))
        i2 = torch.cat([c1,self.upsample2(d1)], dim=1)
        d2 = F.relu(self.dec_conv2(i2))
        i3 = torch.cat([c0,self.upsample3(d2)], dim=1)
        d3 = self.dec_conv3(i3)  # no activation
        return d3

class UNet2(nn.Module):
    def __init__(self):
        super().__init__()

        # encoder (downsampling)
        self.enc_conv0 = nn.Conv2d(3, 64, 3, padding=1)
        self.downsample0 = nn.Conv2d(64, 64, 2, stride=2, padding=0)  # 128 -> 64
        self.enc_conv1 = nn.Conv2d(64, 64, 3, padding=1)
        self.downsample1 = nn.Conv2d(64, 64, 2, stride=2, padding=0)  # 64 -> 32
        self.enc_conv2 = nn.Conv2d(64, 64, 3, padding=1)
        self.downsample2 = nn.Conv2d(64, 64, 2, stride=2, padding=0)  # 32 -> 16
        self.enc_conv3 = nn.Conv2d(64, 64, 3, padding=1)
        self.downsample3 = nn.Conv2d(64, 64, 2, stride=2, padding=0)  # 16 -> 8

        # bottleneck
        self.bottleneck_conv = nn.Conv2d(64, 64, 3, padding=1)

        # decoder (upsampling)
        self.upsample0 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)  # 8 -> 16
        self.dec_conv0 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample1 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)  # 16 -> 32
        self.dec_conv1 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample2 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)  # 32 -> 64
        self.dec_conv2 = nn.Conv2d(64*2, 64, 3, padding=1)
        self.upsample3 = nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0)  # 64 -> 128
        self.dec_conv3 = nn.Conv2d(64*2, 1, 3, padding=1)

    def forward(self, x):
        # encoder
        c0 = F.relu(self.enc_conv0(x))
        e0 = self.downsample0(c0)
        c1 = F.relu(self.enc_conv1(e0))
        e1 = self.downsample1(c1)
        c2 = F.relu(self.enc_conv2(e1))
        e2 = self.downsample2(c2)
        c3 = F.relu(self.enc_conv3(e2))
        e3 = self.downsample3(c3)

        # bottleneck
        b = F.relu(self.bottleneck_conv(e3))

        # decoder
        i0 = torch.cat([c3,self.upsample0(b)], dim=1)
        d0 = F.relu(self.dec_conv0(i0))
        i1 = torch.cat([c2,self.upsample1(d0)], dim=1)
        d1 = F.relu(self.dec_conv1(i1))
        i2 = torch.cat([c1,self.upsample2(d1)], dim=1)
        d2 = F.relu(self.dec_conv2(i2))
        i3 = torch.cat([c0,self.upsample3(d2)], dim=1)
        d3 = self.dec_conv3(i3)  # no activation
        return d3
