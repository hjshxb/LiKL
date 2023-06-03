
import torch
import torch.nn as nn
import math
from torch.nn import functional as F
from typing import Optional
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation as SElayer
from torchvision.ops import (DeformConv2d,
                             deform_conv2d)


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def init_weight(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
        nn.init.ones_(m.weight)
        nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.02)
        # nn.init.normal_(m.weight, 0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


class DepthwiseSeparableConv(nn.Module):
    """ Depthwise separable convolution module.
    Described in  https://arxiv.org/pdf/1704.04861.pdf
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=None,
                 dilation=1,
                 norm_layer=nn.BatchNorm2d,
                 activation_layer=nn.ReLU,
                 last_act=True
                 ) -> None:
        super().__init__()
        # depthwise conv
        self.depthwise_conv = ConvNormActivation(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,
            dilation=dilation,
            norm_layer=norm_layer,
            activation_layer=activation_layer)
        # pointwise conv
        self.pointwise_conv = ConvNormActivation(
            in_channels,
            out_channels,
            kernel_size=1,
            norm_layer=norm_layer,
            activation_layer=activation_layer if last_act else None)

    def forward(self, x):
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        return x


class SkipUpBlock(nn.Module):
    def __init__(self, in_channels1, in_channels2,
                 out_channels1, out_channels2,
                 upsample=True, use_deconv=False,
                 activation_layer=nn.ReLU):
        super().__init__()
        self.conv1 = nn.Sequential()
        if use_deconv and upsample:
            self.conv1 =nn.Sequential(
                nn.ConvTranspose2d(in_channels1,  out_channels1,
                                   kernel_size=2, stride=2, bias=False),
                nn.BatchNorm2d(out_channels1),
                activation_layer(inplace=True))
        else:
            self.conv1.append(ConvNormActivation(
                in_channels1,
                out_channels1,
                kernel_size=1,
                activation_layer=activation_layer)
            )
        
        self.conv2 = nn.Sequential()
        if in_channels2 != out_channels2:
            self.conv2.append(ConvNormActivation(
                in_channels2,
                out_channels2,
                kernel_size=1,
                activation_layer=activation_layer)
            )
        self.upsample = upsample
        self.use_deconv = use_deconv
        self.act = activation_layer(inplace=True)

    def forward(self, x1, x2):
        if (not self.use_deconv) and self.upsample:
            x1 = F.interpolate(x1, scale_factor=2.0,
                               mode='bilinear', align_corners=True)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        out = torch.cat((x1, x2), dim=1)
        # out = x1 + x2
        return out


class StageBlock(nn.Module):
    def __init__(self, in_c, out_c, activation_layer=nn.ReLU):
        super(StageBlock, self).__init__()
        self.conv1 = ConvNormActivation(
            in_c, out_c, kernel_size=3, activation_layer=activation_layer)

        squeeze_channels = _make_divisible(out_c // 2, 8)
        self.attention = SElayer(
            out_c, squeeze_channels, scale_activation=nn.Hardsigmoid)


    def forward(self, x):
        x = self.conv1(x)
        x = self.attention(x)
        return x


class DCNv2(DeformConv2d):
    """
    Performs Deformable Convolution v2, described in https://arxiv.org/abs/1811.11168
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 bias=True) -> None:
        super().__init__(in_channels, out_channels, kernel_size,
                         stride, padding, dilation, groups, bias)

        # offset_mask
        channels_ = self.groups * 3 * self.kernel_size[0] * self.kernel_size[1]
        self.conv_offset_mask = nn.Conv2d(in_channels,
                                          channels_,
                                          kernel_size=self.kernel_size,
                                          stride=self.stride,
                                          padding=self.padding,
                                          bias=False)
        self.init_offset()

    def init_offset(self):
        self.conv_offset_mask.weight.data.zero_()
        if self.conv_offset_mask.bias is not None:
            self.conv_offset_mask.bias.data.zero_()

    def forward(self, input):
        out = self.conv_offset_mask(input)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)
        return deform_conv2d(input,
                             offset,
                             self.weight,
                             self.bias,
                             self.stride,
                             self.padding,
                             self.dilation,
                             mask)

# class MCM(nn.Module):
#     """ Mixture Convolution Module
#     """
#     def __init__(self, in_ch, out_ch, activation_layer = nn.ReLU) -> None:
#         super().__init__()
#         self.use_res = True if in_ch == out_ch else False
#         self.conv1= nn.Sequential(
#             DCNv2(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(in_ch),
#             activation_layer(inplace=True))
#         # self.conv1 = ConvNormActivation(in_ch, in_ch, kernel_size=3, activation_layer=activation_layer)
#         self.conv2 = ConvNormActivation(
#                     in_ch,
#                     in_ch,
#                     kernel_size=3,
#                     padding=2,
#                     dilation=2,
#                     activation_layer=activation_layer)
#         self.conv3 = ConvNormActivation(
#                     in_ch,
#                     out_ch,
#                     kernel_size=3,
#                     padding=3,
#                     dilation=3,
#                     activation_layer=None)
#         self.act = activation_layer(inplace=True)


#     def forward(self, x):
#         x = self.conv1(x)
#         out = self.conv2(x)
#         out = self.conv3(out)
#         if self.use_res:
#             out = out + x
#         return self.act(out)

#     def _init_weight(self):
#         self.apply(init_weight)
#         self.conv1[0].reset_parameters()
#         self.conv1[0].init_offset()

class MCM(nn.Module):
    """ Mixture Convolution Module
    """

    def __init__(self, in_ch, out_ch, activation_layer=nn.ReLU) -> None:
        super().__init__()
        mix_kernel = [
            #s1, s2, d
            [1, 7, 1],
            [3, 3, 1],
            [3, 3, 3],
            [7, 1, 1]]
        self.conv1 = nn.Sequential(
            DCNv2(in_ch, in_ch, kernel_size=3,
                  stride=1, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            activation_layer(inplace=True))
        self.mix_convs = nn.ModuleList(
            [nn.Conv2d(in_ch, in_ch, (s1, s2), 1, ((s1-1)//2*d, (s2-1)//2*d), d, bias=False)
             for s1, s2, d in mix_kernel]
        )
        self.bn = nn.BatchNorm2d(in_ch * len(mix_kernel))
        self.act = activation_layer(inplace=True)
        self.conv2 = ConvNormActivation(
            in_ch * len(mix_kernel), out_ch, kernel_size=1, activation_layer=activation_layer)

    def forward(self, x):
        x = self.conv1(x)
        out = torch.cat([conv(x) for conv in self.mix_convs], dim=1)
        out = self.act(self.bn(out))
        out = self.conv2(out)
        return out

    def _init_weight(self):
        self.apply(init_weight)
        self.conv1[0].reset_parameters()
        self.conv1[0].init_offset()


class AsppModule(nn.Module):
    """
    AsppModule from https://arxiv.org/abs/1802.02611
    """

    def __init__(self, in_channels, out_channels, output_stride=16, norm_layer=nn.BatchNorm2d, activation_layer=nn.ReLU):
        super(AsppModule, self).__init__()

        mid_channels = _make_divisible(in_channels, 8)
        # output_stride choice
        if output_stride == 16:
            atrous_rates = [0, 6, 12, 18]
        elif output_stride == 8:
            atrous_rates = [0, 12, 24, 36]
        else:
            raise Warning("output_stride must be 8 or 16!")
        # atrous_spatial_pyramid_pooling part
        self._atrous_convolution1 = ConvNormActivation(in_channels, mid_channels, 1, 1,
                                                       dilation=1, norm_layer=norm_layer, activation_layer=activation_layer)
        self._atrous_convolution2 = ConvNormActivation(in_channels, mid_channels, 3, 1,
                                                       dilation=atrous_rates[1], norm_layer=norm_layer, activation_layer=activation_layer)
        self._atrous_convolution3 = ConvNormActivation(in_channels, mid_channels, 3, 1,
                                                       dilation=atrous_rates[2], norm_layer=norm_layer, activation_layer=activation_layer)
        self._atrous_convolution4 = ConvNormActivation(in_channels, mid_channels, 3, 1,
                                                       dilation=atrous_rates[3], norm_layer=norm_layer, activation_layer=activation_layer)

        # image_pooling part
        self._image_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            ConvNormActivation(in_channels, mid_channels,
                               kernel_size=1, norm_layer=norm_layer, activation_layer=activation_layer)
        )

        # last_conv
        self.last_conv = ConvNormActivation(mid_channels * 5, out_channels, 1,
                                            norm_layer=norm_layer, activation_layer=activation_layer)

    def forward(self, x):
        x1 = self._atrous_convolution1(x)
        x2 = self._atrous_convolution2(x)
        x3 = self._atrous_convolution3(x)
        x4 = self._atrous_convolution4(x)
        x5 = self._image_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[
                           2:4], mode='bilinear', align_corners=True)
        cat_out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        cat_out = self.last_conv(cat_out)
        return cat_out


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class PositionalEncodingFourier(nn.Module):
    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * math.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(
            self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)

        return pos
