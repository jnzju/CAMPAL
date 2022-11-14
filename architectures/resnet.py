import torch
from torch import Tensor
import torch.nn as nn
from typing import Type, Any, Callable, Union, List, Optional
from .builder import MODELS


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth',
}


def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
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
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

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
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

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
        drop_ratio=0.0,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.dropout = torch.nn.Dropout(p=drop_ratio)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
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
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        out1 = self.layer1(x)
        out1 = self.dropout(out1)
        out2 = self.layer2(out1)
        out2 = self.dropout(out2)
        out3 = self.layer3(out2)
        out3 = self.dropout(out3)
        out4 = self.layer4(out3)
        out4 = self.dropout(out4)

        outf = self.avgpool(out4)
        outf = torch.flatten(outf, 1)
        out = self.fc(outf)

        return out, outf, [out1, out2, out3, out4]

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(
    arch: str,
    block: Type[Union[BasicBlock, Bottleneck]],
    layers: List[int],
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model


@MODELS.register_module()
class resnet18(object):
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnet34(object):
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnet50(object):
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnet101(object):
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnet152(object):
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnext50_32x4d(object):
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:

        kwargs['groups'] = 32
        kwargs['width_per_group'] = 4
        return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class resnext101_32x8d(object):
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        kwargs['groups'] = 32
        kwargs['width_per_group'] = 8
        return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class wide_resnet50_2(object):
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        kwargs['width_per_group'] = 64 * 2
        return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3], self.pretrained, self.progress,
                       **kwargs)


@MODELS.register_module()
class wide_resnet101_2(object):
    def __init__(self, pretrained: bool = False, progress: bool = True):
        self.pretrained = pretrained
        self.progress = progress

    def __call__(self, **kwargs) -> ResNet:
        kwargs['width_per_group'] = 64 * 2
        return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3], self.pretrained, self.progress,
                       **kwargs)
