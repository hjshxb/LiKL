"""
reference from torchvison/models/mobilenetv3.py
"""

import torch.nn as nn
import math
from torch import Tensor
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation as SElayer
from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Any, Callable, List, Optional, Sequence
from functools import partial
from .attentions import CBAM
from .layers import _make_divisible, AsppModule

__all__ = ["MobileNetV2", "mobilenet_v2"]


model_urls = {
    "mobilenet_v2": "https://download.pytorch.org/models/mobilenet_v2-b0353104.pth",
}


class SGBlock(nn.Module):
    """
    Sandglass block from MobileNeXt(http://arxiv.org/abs/2007.02269)
    Code from https://github.com/murufeng/awesome_lightweight_networks/
    """

    def __init__(self, inp, oup, stride, expand_ratio, dilation: int = 1, keep_3x3=False):
        super(SGBlock, self).__init__()
        self.stride = 1 if dilation > 1 else stride
        assert stride in [1, 2]

        hidden_dim = inp // expand_ratio
        if hidden_dim < oup / 6.:
            hidden_dim = math.ceil(oup / 6.)
            hidden_dim = _make_divisible(hidden_dim, 16)  # + 16

        # self.relu = nn.ReLU6(inplace=True)
        self.identity = False
        self.identity_div = 1
        self.expand_ratio = expand_ratio
        if expand_ratio == 2:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, self.stride, 1 * dilation,
                          dilation=dilation, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        elif inp != oup and stride == 1 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
            )
        elif inp != oup and stride == 2 and keep_3x3 == False:
            self.conv = nn.Sequential(
                # pw-linear
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, self.stride, 1 * dilation,
                          dilation=dilation, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            if keep_3x3 == False:
                self.identity = True
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(inp, inp, 3, 1, 1 * dilation,
                          dilation=dilation, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                # nn.ReLU6(inplace=True),
                # pw
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(oup, oup, 3, 1, 1 * dilation,
                          dilation=dilation, groups=oup, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        out = self.conv(x)

        if self.identity:
            return out + x
        else:
            return out


class InvertedResidual(nn.Module):
    def __init__(
        self, inp: int, oup: int, stride: int, expand_ratio: int, dilation: int = 1,
        use_attention: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        attention_layer: Callable[..., nn.Module] = partial(
            SElayer, scale_activation=nn.Hardsigmoid)
    ) -> None:
        super().__init__()
        self.stride = 1 if dilation > 1 else stride
        assert stride in [1, 2]

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = nn.ReLU6

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pw
            layers.append(
                ConvNormActivation(inp, hidden_dim, kernel_size=1,
                                   norm_layer=norm_layer, activation_layer=activation_layer)
            )
        layers.append(
            # dw
            ConvNormActivation(
                hidden_dim,
                hidden_dim,
                stride=stride,
                groups=hidden_dim,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
                dilation=dilation
            ))

        self.use_attention = use_attention
        if use_attention:
            squeeze_channels = _make_divisible(hidden_dim // 4, 8)
            # layers.append(attention_layer(hidden_dim, squeeze_channels))
            self.attention = attention_layer(hidden_dim, squeeze_channels)

        layers.extend(
            [
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                norm_layer(oup)]
        )

        self.conv = nn.Sequential(*layers)
        self.out_channels = oup
        self._is_cn = stride > 1

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv[0](x)
        out = self.conv[1](out)
        if self.use_attention:
            out = self.attention(out)
        if self.use_res_connect:
            return x + self.conv[2](out)
        else:
            return self.conv[2](out)


class MobileNetV2(nn.Module):
    def __init__(
        self,
        fpn_selected: List[int],
        fpn_channel_list: List[int],
        input_channel: int = 3,
        inverted_residual_setting: Optional[List[List[int]]] = None,
        round_nearest: int = 8,
        block: Optional[Callable[..., nn.Module]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        activation_layer: Optional[Callable[..., nn.Module]] = None,
        use_aspp: bool = False,
    ) -> None:
        """
        MobileNet V2 main class

        Args:
            num_classes (int): Number of classes
            width_mult (float): Width multiplier - adjusts number of channels in each layer by this amount
            inverted_residual_setting: Network structure
            round_nearest (int): Round the number of channels in each layer to be a multiple of this number
            Set to 1 to turn off rounding
            block: Module specifying inverted residual building block for mobilenet
            norm_layer: Module specifying the normalization layer to use
            dropout (float): The droupout probability

        """
        super().__init__()

        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = nn.BatchNorm2d

        if activation_layer is None:
            activation_layer = nn.ReLU6

        self.use_aspp = use_aspp
        width_mult = 1.0
        mid_channel = 32

        if inverted_residual_setting is None:
            if block is InvertedResidual:
                inverted_residual_setting = [
                    # t, c, n, s, dilation, attention
                    [1, 16, 1, 1, 1, False],
                    [6, 24, 2, 2, 1, False],  # /4
                    [6, 32, 3, 2, 1, False],  # /8
                    [6, 64, 4, 2, 1, False],  # /16
                    [6, 96, 3, 1, 1, False],
                    [6, 160, 3, 2, 1, False],  # /32
                    [6, 320, 1, 1, 1, False],
                ]
            else:
                raise ValueError("Block can't be {}".format(block.__name__))

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 6:
            raise ValueError(
                f"inverted_residual_setting should be non-empty or a 5-element list, got {inverted_residual_setting}"
            )

        # building first layer
        mid_channel = _make_divisible(mid_channel * width_mult, round_nearest)
        features: List[nn.Module] = [
            ConvNormActivation(input_channel, mid_channel, stride=2,
                               norm_layer=norm_layer, activation_layer=activation_layer)
        ]

        # building inverted residual blocks
        for t, c, n, s, d, use_attention in inverted_residual_setting:
            output_channel = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                if block is InvertedResidual:
                    features.append(block(mid_channel, output_channel, stride, expand_ratio=t,
                                          dilation=d, norm_layer=norm_layer, activation_layer=activation_layer, use_attention=use_attention))
                else:
                    features.append(block(mid_channel, output_channel, stride,
                                    expand_ratio=t, dilation=d, keep_3x3=(s == 1 and n == 1)))
                mid_channel = output_channel

        # make it nn.Sequential
        self.features = nn.Sequential(*features)

        if fpn_selected[-1] > len(features):
            raise ValueError(
                "The fpn_selected layer should less than {}".format(len(features)))
        else:
            self.fpn_selected = fpn_selected
            self.fpn_channel_list = fpn_channel_list

        # aspp module
        if use_aspp:
            self.aspp = AsppModule(
                mid_channel, mid_channel, 16, norm_layer=norm_layer, activation_layer=activation_layer)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # This exists since TorchScript doesn't support inheritance, so the superclass method
        # (this one) needs to have a name other than `forward` that can be accessed in a subclass
        features = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i == len(self.features) - 1 and self.use_aspp:
                x = self.aspp(x)
            if i in self.fpn_selected:
                features.append(x)
        return features

    def forward(self, x: Tensor):
        return self._forward_impl(x)


def mobilenet_v2(arch="mobilenet_v2", pretrained: bool = False, input_channel: int = 3,
                 progress: bool = True, **kwargs: Any) -> MobileNetV2:
    """
    Constructs a MobileNetV2 architecture from
    `"MobileNetV2: Inverted Residuals and Linear Bottlenecks" <https://arxiv.org/abs/1801.04381>`_.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    block = InvertedResidual
    fpn_selected = [3, 6, 13, 17]
    fpn_channel_list = [24, 32, 96, 320]

    model = MobileNetV2(fpn_selected, fpn_channel_list,
                        input_channel, block=block, **kwargs)
    if pretrained:
        if model_urls.get(arch, None) is None:
            raise ValueError(
                f"No checkpoint is available for model type {arch}")
        pretrained_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress)
        state_dict = model.state_dict()
        model_dict = {}
        for k, v in pretrained_dict.items():
            if input_channel == 1 and k.startswith("features.0.0"):
                continue
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model
