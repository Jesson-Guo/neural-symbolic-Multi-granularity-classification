from typing import Any, Callable, List, Optional, Type, Union
import math

import torch
import torch.nn as nn
from torch import Tensor

from model.attention.CBAM import CBAMBlock


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self, 
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAMBlock(channel=planes,reduction=16,kernel_size=7)
        else:
            self.cbam = None
    
    def forward(self, x: Tensor) -> Tensor:
        # identity: 残差
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # apply CBAM on the convolution outputs in each block
        if not self.cbam is None:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        if use_cbam:
            self.cbam = CBAMBlock(channel=planes*4,reduction=16,kernel_size=7)
        else:
            self.cbam = None
    
    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        # apply CBAM on the convolution outputs in each block
        if not self.cbam is None:
            out = self.cbam(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        use_cbam: bool = False,
    ) -> None:
        super().__init__()
        # !!!!不同的数据集对应的参数不相同, 当前配置适用于ImageNet
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                f"or a 3-element tuple, got {replace_stride_with_dilation}"
            )
        self.groups = groups
        self.base_width = width_per_group

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], use_cbam=use_cbam)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0], use_cbam=use_cbam)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1], use_cbam=use_cbam)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2], use_cbam=use_cbam)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

        '''
        init.kaiming_normal(self.fc.weight)
        for key in self.state_dict():
            if key.split('.')[-1]=="weight":
                if "conv" in key:
                    init.kaiming_normal(self.state_dict()[key], mode='fan_out')
                if "bn" in key:
                    if "SpatialGate" in key:
                        self.state_dict()[key][...] = 0
                    else:
                        self.state_dict()[key][...] = 1
            elif key.split(".")[-1]=='bias':
                self.state_dict()[key][...] = 0
        '''
    
    def _make_layer(
        self,
        block: Type[Union[BasicBlock, Bottleneck]],
        planes: int,
        blocks: int,
        stride: int = 1,
        dilate: bool = False,
        use_cbam: bool = False,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes, planes, stride, downsample, self.groups, 
                self.base_width, previous_dilation, norm_layer, use_cbam
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                    use_cbam=use_cbam
                )
            )

        return nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        # ImageNet 需要maxpool
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # ImageNet 需要maxpool
        x = self.avgpool(x)
        # F.avg_pool2d(x, 4)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# class ResNetFPN(nn.Module):
#     def __init__(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         layers: List[int],
#         num_classes: int = 1000,
#         scale=1,
#         use_cbam: bool = False,
#     ) -> None:
#         super().__init__()

#         self.inplanes = 64
#         self.pyramid_channels = 256

#         self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
#         self.bn1 = nn.BatchNorm2d(self.inplanes)
#         self.relu1 = nn.ReLU(inplace=True)
#         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

#         self.layer1 = self._make_layer(block, 64, layers[0], use_cbam=use_cbam)
#         self.layer2 = self._make_layer(block, 128, layers[1], stride=2, use_cbam=use_cbam)
#         self.layer3 = self._make_layer(block, 256, layers[2], stride=2, use_cbam=use_cbam)
#         self.layer4 = self._make_layer(block, 512, layers[3], stride=2, use_cbam=use_cbam)
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         # regress to output: [x_1, y_1, x_2, y_2]
#         # self.rpn_head = nn.Linear(self.pyramid_channels, 4)
#         self.rpn_head = nn.Sequential(
#             conv3x3(self.pyramid_channels, )
#         )

#         # top layers
#         self.toplayer = nn.Conv2d(2048, 256, kernel_size=1, stride=1, padding=0)  # Reduce channels
#         self.toplayer_bn = nn.BatchNorm2d(256)
#         self.toplayer_relu = nn.ReLU(inplace=True)

#         # Smooth layers
#         self.smooth1 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth1_bn = nn.BatchNorm2d(256)
#         self.smooth1_relu = nn.ReLU(inplace=True)

#         self.smooth2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth2_bn = nn.BatchNorm2d(256)
#         self.smooth2_relu = nn.ReLU(inplace=True)

#         self.smooth3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
#         self.smooth3_bn = nn.BatchNorm2d(256)
#         self.smooth3_relu = nn.ReLU(inplace=True)

#         # Lateral layers
#         self.latlayer1 = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer1_bn = nn.BatchNorm2d(256)
#         self.latlayer1_relu = nn.ReLU(inplace=True)

#         self.latlayer2 = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer2_bn = nn.BatchNorm2d(256)
#         self.latlayer2_relu = nn.ReLU(inplace=True)

#         self.latlayer3 = nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)
#         self.latlayer3_bn = nn.BatchNorm2d(256)
#         self.latlayer3_relu = nn.ReLU(inplace=True)

#         self.conv2 = nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(256)
#         self.relu2 = nn.ReLU(inplace=True)
#         self.conv3 = nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0)

#         self.scale = scale

#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
#                 m.weight.data.normal_(0, math.sqrt(2. / n))
#             elif isinstance(m, nn.BatchNorm2d):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()

#     def _make_layer(
#         self,
#         block: Type[Union[BasicBlock, Bottleneck]],
#         planes: int,
#         blocks: int,
#         stride: int = 1,
#         use_cbam: bool = False,
#     ) -> nn.Sequential:
#         downsample = None
#         if stride != 1 or self.inplanes != planes * block.expansion:
#             downsample = nn.Sequential(
#                 conv1x1(self.inplanes, planes * block.expansion, stride),
#                 nn.BatchNorm2d(planes * block.expansion),
#             )

#         layers = []
#         layers.append(
#             block(self.inplanes, planes, stride, downsample, use_cbam=use_cbam)
#         )
#         self.inplanes = planes * block.expansion
#         for _ in range(1, blocks):
#             layers.append(
#                 block(self.inplanes, planes, use_cbam=use_cbam)
#             )

#         return nn.Sequential(*layers)
    
#     def _upsample(self, x, y, scale=1):
#         _, _, H, W = y.size()
#         return nn.functional.upsample(x, size=(H // scale, W // scale), mode='bilinear')

#     def _upsample_add(self, x, y):
#         _, _, H, W = y.size()
#         return nn.functional.upsample(x, size=(H, W), mode='bilinear') + y

#     def forward(self, x: Tensor) -> Tensor:
#         x = self.conv1(x)
#         x = self.bn1(x)
#         x = self.relu1(x)
#         x = self.maxpool(x)

#         x = self.layer1(x)
#         c2 = x
#         x = self.layer2(x)
#         c3 = x
#         x = self.layer3(x)
#         c4 = x
#         x = self.layer4(x)
#         c5 = x

#         # Top-down
#         p5 = self.toplayer(c5)
#         p5 = self.toplayer_relu(self.toplayer_bn(p5))

#         c4 = self.latlayer1(c4)
#         c4 = self.latlayer1_relu(self.latlayer1_bn(c4))
#         p4 = self._upsample_add(p5, c4)
#         p4 = self.smooth1(p4)
#         p4 = self.smooth1_relu(self.smooth1_bn(p4))

#         c3 = self.latlayer2(c3)
#         c3 = self.latlayer2_relu(self.latlayer2_bn(c3))
#         p3 = self._upsample_add(p4, c3)
#         p3 = self.smooth2(p3)
#         p3 = self.smooth2_relu(self.smooth2_bn(p3))

#         c2 = self.latlayer3(c2)
#         c2 = self.latlayer3_relu(self.latlayer3_bn(c2))
#         p2 = self._upsample_add(p3, c2)
#         p2 = self.smooth3(p2)
#         p2 = self.smooth3_relu(self.smooth3_bn(p2))

#         p3 = self._upsample(p3, p2)
#         p4 = self._upsample(p4, p2)
#         p5 = self._upsample(p5, p2)

#         p5 = self.avgpool(p5)

#         boundary_box = 

#         # x = torch.cat((p2, p3, p4, p5), 1)
#         # x = self.conv2(x)
#         # x = self.relu2(self.bn2(x))
#         # x = self.conv3(x)
#         # x = self._upsample(x, x, scale=self.scale)

#         return x


def resnet18(use_cbam) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], use_cbam=use_cbam)


def resnet34(use_cbam) -> ResNet:
    return ResNet(BasicBlock, [3, 4, 6, 3], use_cbam=use_cbam)


def resnet50(use_cbam) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 6, 3], use_cbam=use_cbam)


def resnet101(use_cbam) -> ResNet:
    return ResNet(Bottleneck, [3, 4, 23, 3], use_cbam=use_cbam)


def resnet152(use_cbam) -> ResNet:
    return ResNet(Bottleneck, [3, 8, 36, 3], use_cbam=use_cbam)
