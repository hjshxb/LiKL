import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.misc import ConvNormActivation
from .layers import (MCM, SkipUpBlock,
                     StageBlock,
                     DCNv2,
                     init_weight)


class FpnDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, use_deconv=True, activation_layer=nn.ReLU) -> None:
        super().__init__()
        in_channels_list = in_channels_list[::-1]
        blocks = []
        for i in range(len(out_channels_list)):
            if i == 0:
                b = nn.ModuleList([
                    SkipUpBlock(
                        in_channels_list[i],
                        in_channels_list[i+1],
                        out_channels_list[i],
                        out_channels_list[i] // 2,
                        upsample=True,
                        use_deconv=use_deconv,
                        activation_layer=activation_layer),
                    StageBlock(
                        out_channels_list[i] + out_channels_list[i] // 2, 
                        out_channels_list[i], activation_layer)]
                )
            else:
                b = nn.ModuleList([
                    SkipUpBlock(
                        out_channels_list[i-1],
                        in_channels_list[i+1],
                        out_channels_list[i],
                        out_channels_list[i] // 2,
                        upsample=True,
                        use_deconv=use_deconv,
                        activation_layer=activation_layer),
                    StageBlock(
                        out_channels_list[i] + out_channels_list[i] // 2, 
                        out_channels_list[i], activation_layer)]
                )
            blocks.append(b)
        self.blocks = nn.Sequential(*blocks)

        # self.last_conv = nn.Sequential(
        #     DCNv2(out_channels_list[-1], out_channels_list[-1], kernel_size=3, stride=1, padding=1, bias=False),
        #     nn.BatchNorm2d(out_channels_list[-1]),
        #     activation_layer(inplace=True))

    def forward(self, feats):
        x = feats[-1]
        for i, block in enumerate(self.blocks):
            x = block[0](x, feats[-(i + 2)])
            x = block[1](x)
        # x = self.last_conv(x)
        return x

    def _init_weight(self):
        self.apply(init_weight)
        # self.last_conv[0].reset_parameters()
        # self.last_conv[0].init_offset()


class FusedDecoder(nn.Module):
    def __init__(self, in_channels_list, out_channels, activation_layer=nn.ReLU) -> None:
        super().__init__()
        up_scale = [2, 4, 8]
        blocks = [ConvNormActivation(
            in_ch, out_channels,
            kernel_size=1,
            activation_layer=None)
            for in_ch in in_channels_list]
        self.up_blocks = nn.ModuleList([
            nn.Upsample(scale_factor=s, mode="bilinear", align_corners=True)
            for s in up_scale])

        self.blocks = nn.ModuleList(blocks)
        self.act = activation_layer(inplace=True)
        self.last_conv = nn.Sequential(
            DCNv2(out_channels, out_channels, kernel_size=3,
                  stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            activation_layer(inplace=True))

    def forward(self, feats):
        x1 = self.blocks[0](feats[0])
        x2 = self.up_blocks[0](self.blocks[1](feats[1]))
        x3 = self.up_blocks[1](self.blocks[2](feats[2]))
        x4 = self.up_blocks[2](self.blocks[3](feats[3]))
        x = self.last_conv(self.act(x1+x2+x3+x4))
        return x

    def _init_weight(self):
        self.apply(init_weight)
        self.last_conv[0].reset_parameters()
        self.last_conv[0].init_offset()


class LineDecoder(nn.Module):
    def __init__(self, input_feat_dim, out_channels, activation_layer=nn.ReLU) -> None:
        super().__init__()
        in_ch = input_feat_dim
        out_ch = out_channels
        self.mcm = MCM(in_ch, in_ch * 2, activation_layer)
        # self.upconv = nn.Sequential(
        #     nn.ConvTranspose2d(in_ch, in_ch // 2, kernel_size=2, stride=2, bias=False),
        #     nn.BatchNorm2d(in_ch // 2),
        #     activation_layer(inplace=True)
        # )
        self.upconv = nn.PixelShuffle(upscale_factor=2)
        self.branch1 = nn.Sequential(
            ConvNormActivation(in_ch // 2, in_ch // 2,
                               kernel_size=3, activation_layer=activation_layer),
            nn.Conv2d(in_ch // 2, 1, kernel_size=1, padding=0),
        )
        self.branch2 = nn.Sequential(
            ConvNormActivation(in_ch // 2, in_ch // 2,
                               kernel_size=3, activation_layer=activation_layer),
            nn.Conv2d(in_ch // 2, out_ch - 1, kernel_size=1, padding=0)
        )

    def forward(self, x):
        x = self.mcm(x)
        x = self.upconv(x)
        # x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat([x1, x2], dim=1)
        return out

    def _init_weight(self):
        self.apply(init_weight)
        self.mcm._init_weight()


class PointsDecoder(nn.Module):
    def __init__(self, input_feat_dim=64, out_channels=3, activation_layer=nn.ReLU) -> None:
        super().__init__()
        self.conv1 = ConvNormActivation(
            input_feat_dim, input_feat_dim * 2,
            kernel_size=3, stride=2,
            activation_layer=activation_layer)
        self.score_branch = nn.Sequential(
            ConvNormActivation(input_feat_dim * 2, input_feat_dim * 2,
                               kernel_size=3, stride=1,
                               activation_layer=activation_layer),
            nn.Conv2d(input_feat_dim * 2, 1, kernel_size=1))

        self.position_branch = nn.Sequential(
            ConvNormActivation(input_feat_dim * 2, input_feat_dim * 2,
                               kernel_size=3, stride=1,
                               activation_layer=activation_layer),
            nn.Conv2d(input_feat_dim * 2, 2, kernel_size=1))

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.score_branch(x)
        x2 = self.position_branch(x)
        out = torch.cat([x1, x2], dim=1)
        return out


class DescriptorDecoder(nn.Module):
    def __init__(self, input_feat_dim=64, out_channels=128, activation_layer=nn.ReLU) -> None:
        super().__init__()
        self.conv1 = ConvNormActivation(
            input_feat_dim,
            input_feat_dim,
            stride=1,
            kernel_size=3,
            activation_layer=activation_layer)
        self.conv2 = nn.Conv2d(input_feat_dim, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x
