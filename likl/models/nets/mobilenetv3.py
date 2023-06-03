"""
reference from torchvison/models/mobilenetv3.py
"""
import torch
import torch.nn as nn
from torch import Tensor
from torchvision.ops.misc import ConvNormActivation, SqueezeExcitation as SElayer
from torchvision._internally_replaced_utils import load_state_dict_from_url
from typing import Any, Callable, List, Optional, Sequence
from functools import partial
# from torchvision.models.feature_extraction import get_graph_node_names
# from torchvision.models.feature_extraction import create_feature_extractor
from .layers import _make_divisible, AsppModule


__all__ = ["MobilenetV3", "mobilenet_v3"]
model_urls = {
    "mobilenet_v3_large": "https://download.pytorch.org/models/mobilenet_v3_large-8738ca79.pth",
    "mobilenet_v3_small": "https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth",
}


class InvertedResidualConfig:
    # Stores information listed at Tables 1 and 2 of the MobileNetV3 paper
    def __init__(
        self,
        input_channels: int,
        kernel: int,
        expanded_channels: int,
        out_channels: int,
        use_se: bool,
        activation: str,
        stride: int,
        dilation: int,
        width_mult: float,
    ):
        self.input_channels = self.adjust_channels(input_channels, width_mult)
        self.kernel = kernel
        self.expanded_channels = self.adjust_channels(
            expanded_channels, width_mult)
        self.out_channels = self.adjust_channels(out_channels, width_mult)
        self.use_se = use_se
        self.use_hs = activation == "HS"
        self.stride = stride
        self.dilation = dilation

    @staticmethod
    def adjust_channels(channels: int, width_mult: float):
        return _make_divisible(channels * width_mult, 8)


class InvertedResidual(nn.Module):
    # Implemented as described at section 5 of MobileNetV3 paper
    def __init__(
        self,
        cnf: InvertedResidualConfig,
        norm_layer: Callable[..., nn.Module],
        se_layer: Callable[..., nn.Module] = partial(
            SElayer, scale_activation=nn.Hardsigmoid),
    ):
        super().__init__()
        if not (1 <= cnf.stride <= 2):
            raise ValueError("illegal stride value")

        self.use_res_connect = cnf.stride == 1 and cnf.input_channels == cnf.out_channels

        layers: List[nn.Module] = []
        activation_layer = nn.Hardswish if cnf.use_hs else nn.ReLU

        # expand
        if cnf.expanded_channels != cnf.input_channels:
            layers.append(
                ConvNormActivation(
                    cnf.input_channels,
                    cnf.expanded_channels,
                    kernel_size=1,
                    norm_layer=norm_layer,
                    activation_layer=activation_layer,
                )
            )

        # depthwise
        stride = 1 if cnf.dilation > 1 else cnf.stride
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels,
                cnf.expanded_channels,
                kernel_size=cnf.kernel,
                stride=stride,
                dilation=cnf.dilation,
                groups=cnf.expanded_channels,
                norm_layer=norm_layer,
                activation_layer=activation_layer,
            )
        )
        if cnf.use_se:
            squeeze_channels = _make_divisible(cnf.expanded_channels // 4, 8)
            layers.append(se_layer(cnf.expanded_channels, squeeze_channels))

        # project
        layers.append(
            ConvNormActivation(
                cnf.expanded_channels, cnf.out_channels, kernel_size=1, norm_layer=norm_layer, activation_layer=None
            )
        )

        self.block = nn.Sequential(*layers)
        self.out_channels = cnf.out_channels
        self._is_cn = cnf.stride > 1

    def forward(self, input: Tensor) -> Tensor:
        result = self.block(input)
        if self.use_res_connect:
            result += input
        return result


class MobileNetV3(nn.Module):
    def __init__(
        self,
        inverted_residual_setting: List[InvertedResidualConfig],
        fpn_selected: List[int],
        fpn_channel_list: List[int],
        input_channel: int = 3,
        block: Optional[Callable[..., nn.Module]] = None,
        use_aspp: bool = False,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        **kwargs: Any,
    ) -> None:
        """
        MobileNet V3 main class

        Args:
            inverted_residual_setting (List[InvertedResidualConfig]): Network structure
            input_channel (int): The number of channels on the first layer
            block (Optional[Callable[..., nn.Module]]): Module specifying inverted residual building block for mobilenet
            norm_layer (Optional[Callable[..., nn.Module]]): Module specifying the normalization layer to use
            fpn_selected (List[int]): 
        """
        super().__init__()
        if not inverted_residual_setting:
            raise ValueError(
                "The inverted_residual_setting should not be empty")
        elif not (
            isinstance(inverted_residual_setting, Sequence)
            and all([isinstance(s, InvertedResidualConfig) for s in inverted_residual_setting])
        ):
            raise TypeError(
                "The inverted_residual_setting should be List[InvertedResidualConfig]")
        if block is None:
            block = InvertedResidual

        if norm_layer is None:
            norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.01)

        layers: List[nn.Module] = []
        # building first layer
        firstconv_output_channels = inverted_residual_setting[0].input_channels
        layers.append(
            ConvNormActivation(
                input_channel,
                firstconv_output_channels,
                kernel_size=3,
                stride=2,
                norm_layer=norm_layer,
                activation_layer=nn.Hardswish,
            )
        )

        # building inverted residual blocks
        for cnf in inverted_residual_setting:
            layers.append(block(cnf, norm_layer))

        if fpn_selected[-1] > len(layers):
            raise ValueError(
                "The fpn_selected layer should less than {}".format(len(layers)))
        else:
            self.fpn_selected = fpn_selected
            self.fpn_channel_list = fpn_channel_list
        self.features = nn.Sequential(*layers)

        # aspp module
        self.use_aspp = use_aspp
        if use_aspp:
            self.aspp = AsppModule(inverted_residual_setting[-1].out_channels,
                                   128, 8, norm_layer=norm_layer, activation_layer=nn.Hardswish)

        # init params
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

    def _forward_impl(self, x: Tensor) -> List[Tensor]:
        features = []
        for i, f in enumerate(self.features):
            x = f(x)
            if i == len(self.features) - 1 and self.use_aspp:
                x = self.aspp(x)
            if i in self.fpn_selected:
                features.append(x)
        return features

    def forward(self, x: Tensor) -> List[Tensor]:
        return self._forward_impl(x)


def _mobilenet_v3_conf(
    arch: str, width_mult: float = 1.0, reduced_tail: bool = False, dilated: bool = False, **kwargs: Any
):
    reduce_divider = 2 if reduced_tail else 1
    dilation = 2 if dilated else 1

    bneck_conf = partial(InvertedResidualConfig, width_mult=width_mult)

    if arch == "mobilenet_v3_large":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, False, "RE", 1, 1),
            bneck_conf(16, 3, 64, 24, False, "RE", 2, 1),  # /4
            bneck_conf(24, 3, 72, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 72, 40, True, "RE", 2, 1),  # /8
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 5, 120, 40, True, "RE", 1, 1),
            bneck_conf(40, 3, 240, 80, False, "HS", 2, 1),  # /16
            bneck_conf(80, 3, 200, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 184, 80, False, "HS", 1, 1),
            bneck_conf(80, 3, 480, 112, True, "HS", 1, 1),
            bneck_conf(112, 3, 672, 112, True, "HS", 1, 1),
            bneck_conf(112, 5, 672, 160 // reduce_divider,
                       True, "HS", 2, dilation),  # /32
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider,
                       160 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(160 // reduce_divider, 5, 960 // reduce_divider,
                       160 // reduce_divider, True, "HS", 1, dilation),
        ]
        fpn_selected = [3, 6, 12, 15]
        fpn_channel_list = [24, 40, 112, 160]
    elif arch == "mobilenet_v3_small":
        inverted_residual_setting = [
            bneck_conf(16, 3, 16, 16, True, "RE", 2, 1),  # /4
            bneck_conf(16, 3, 72, 24, False, "RE", 2, 1),  # /8
            bneck_conf(24, 3, 88, 24, False, "RE", 1, 1),
            bneck_conf(24, 5, 96, 40, True, "HS", 2, 1),  # /16
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 240, 40, True, "HS", 1, 1),
            bneck_conf(40, 5, 120, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 144, 48, True, "HS", 1, 1),
            bneck_conf(48, 5, 288, 96 // reduce_divider,
                       True, "HS", 2, dilation),  # /32
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                       96 // reduce_divider, True, "HS", 1, dilation),
            bneck_conf(96 // reduce_divider, 5, 576 // reduce_divider,
                       96 // reduce_divider, True, "HS", 1, dilation),
        ]
        fpn_selected = [0, 1, 3, 8, 11]
    else:
        raise ValueError(f"Unsupported model type {arch}")

    return inverted_residual_setting, fpn_selected, fpn_channel_list


def mobilenet_v3(
    arch: str,
    input_channel: int,
    pretrained: bool,
    progress: bool = False,
    **kwargs: Any,
):
    inverted_residual_setting, fpn_selected, fpn_channel_list = _mobilenet_v3_conf(
        arch, **kwargs)
    model = MobileNetV3(inverted_residual_setting, fpn_selected,
                        fpn_channel_list, input_channel, **kwargs)
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
                print(f"[Info]: Ignore {k} of pretrained model")
                continue
            if k in state_dict:
                model_dict[k] = v
        state_dict.update(model_dict)
        model.load_state_dict(state_dict)
    return model
