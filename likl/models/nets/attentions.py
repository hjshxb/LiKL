import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops.misc import ConvNormActivation
from typing import Callable

from .layers import _make_divisible, DepthwiseSeparableConv


class ChannelAttention(nn.Module):
    def __init__(
        self,
        in_channels: int,
        reduce_ratio: int = 16,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        squeeze_channels = _make_divisible(in_channels // reduce_ratio, 8)
        self.fc = nn.Sequential(
            nn.Conv2d(in_channels, squeeze_channels, 1, bias=False),
            activation(),
            nn.Conv2d(squeeze_channels, in_channels, 1, bias=False)
        )
        self.scale_activation = scale_activation()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.scale_activation(out)


class SpatialAttention(nn.Module):
    def __init__(
        self,
        kernel_size=7,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(
            2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.scale_activaion = scale_activation()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.scale_activaion(x)


class CBAM(nn.Module):
    """Convolutional Block Attention Module
    from https://openaccess.thecvf.com/content_ECCV_2018/html/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.html
    """

    def __init__(
        self,
        in_channels,
        reduce_ratio: int = 16,
        kernel_size: int = 7,
        activation: Callable[..., nn.Module] = nn.ReLU,
        scale_activation: Callable[..., nn.Module] = nn.Sigmoid,
    ):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(
            in_channels, reduce_ratio, activation, scale_activation)
        self.spatial_attention = SpatialAttention(
            kernel_size, scale_activation)

    def forward(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x
        return x


class PRM(nn.Module):
    """Pose Refine Machine
    Described in https://arxiv.org/abs/2003.04030
    """

    def __init__(self, out_channels):
        super().__init__()
        self.out_channels = out_channels
        self.top_path = ConvNormActivation(
            self.out_channels,
            self.out_channels,
            kernel_size=3)
        self.mid_path = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvNormActivation(
                self.out_channels,
                self.out_channels,
                kernel_size=1),
            ConvNormActivation(
                self.out_channels,
                self.out_channels,
                kernel_size=1),
            nn.Sigmoid())
        self.bottom_path = nn.Sequential(
            ConvNormActivation(
                self.out_channels,
                self.out_channels,
                kernel_size=1),
            DepthwiseSeparableConv(
                self.out_channels,
                1,
                kernel_size=9,
                stride=1,
                padding=4),
            nn.Sigmoid())

    def forward(self, x):
        out = self.top_path(x)
        top_out = out
        mid_out = self.mid_path(out)
        bottom_out = self.bottom_path(out)
        out = top_out * (1 + mid_out * bottom_out)

        return out
