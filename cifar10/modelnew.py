import torch
import torch.nn as nn
from layersnew import *
import torch.nn.functional as F
import re
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch.hub import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple
import xlsxwriter
import openpyxl
import os.path

__all__ = ['DenseNet', 'densenet121', 'densenet169', 'densenet201', 'densenet161']

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features: int,
        growth_rate: int,
        bn_size: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseLayer, self).__init__()
        self.spikeA = LIFSpike()
        self.spikeB = LIFSpike()
        self.norm1: tdNorm(newBatchNorm)
        self.add_module('norm1', tdNorm(newBatchNorm(num_input_features)))
        self.spike1: self.spikeA
        self.add_module('spike1', self.spikeA)
        self.conv1: tdConv(nn.Conv2d)
        self.add_module('conv1', tdConv(nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)))
        self.norm2: tdNorm(newBatchNorm)
        self.add_module('norm2', tdNorm(newBatchNorm(bn_size * growth_rate)))
        self.spike2: self.spikeB
        self.add_module('spike2', self.spikeB)
        self.conv2: tdConv(nn.Conv2d)
        self.add_module('conv2', tdConv(nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)))
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs: List[Tensor]) -> Tensor:
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.spike1(self.norm1(concated_features)))
        return bottleneck_output

    # todo: rewrite when torchscript supports any
    def any_requires_grad(self, input: List[Tensor]) -> bool:
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input: List[Tensor]) -> Tensor:
        def closure(*inputs):
            return self.bn_function(inputs)

        return cp.checkpoint(closure, *input)

    @torch.jit._overload_method
    def forward(self, input: List[Tensor]) -> Tensor:
        pass

    @torch.jit._overload_method
    def forward(self, input: Tensor) -> Tensor:
        pass

    def forward(self, input: Tensor) -> Tensor:
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")

            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.spike2(self.norm2(bottleneck_output)))
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers: int,
        num_input_features: int,
        bn_size: int,
        growth_rate: int,
        drop_rate: float,
        memory_efficient: bool = False
    ) -> None:
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features: Tensor) -> Tensor:
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features: int, num_output_features: int) -> None:
        super(_Transition, self).__init__()
        self.spike = LIFSpike()
        self.add_module('norm', tdNorm(newBatchNorm(num_input_features)))
        self.add_module('spike', self.spike)
        self.add_module('conv', tdConv(nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False)))
        self.add_module('pool', tdConv(nn.AvgPool2d(kernel_size=2, stride=2)))


class DenseNet(nn.Module):
    r"""Densenet-BC model class, based on
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        growth_rate (int) - how many filters to add each layer (`k` in paper)
        block_config (list of 4 ints) - how many layers in each pooling block
        num_init_features (int) - the number of filters to learn in the first convolution layer
        bn_size (int) - multiplicative factor for number of bottle neck layers
          (i.e. bn_size * k features in the bottleneck layer)
        drop_rate (float) - dropout rate after each dense layer
        num_classes (int) - number of classification classes
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """

    def __init__(
        self,
        args,
        growth_rate: int = 32,
        block_config: Tuple[int, int, int] = (12, 12, 24),
        num_init_features: int = 64,
        bn_size: int = 4,
        drop_rate: float = 0.2,
        num_classes: int = 10,
        memory_efficient: bool = False,
    ) -> None:

        super(DenseNet, self).__init__()

        self.drop_rate = float(drop_rate)
        self.spike = LIFSpike()
        self.spike1 = LIFSpike()
        self.outspike1 = LIFOutSpike()
        self.outspike2 = LIFOutSpike()
        self.sig = sigOut()
        self.avg = tdConv(nn.AdaptiveAvgPool2d((1, 1)))
        self.features = nn.Sequential(OrderedDict([
            ('conv0', tdConv(nn.Conv2d(3, num_init_features, kernel_size=3, stride=1,
                                padding=1, bias=False))),
            ('norm0', tdNorm(newBatchNorm(num_init_features))),
            ('spike0', self.spike),
            ('pool0', tdConv(nn.Conv2d(num_init_features, num_init_features, kernel_size=3, stride=2,
                                       padding=1, bias=False))),
        ]))

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module('norm5', tdNorm(newBatchNorm(num_features)))

        self.classifier1 = tdConv(nn.Linear(num_features, 256))
        self.classifier2 = tdConv(nn.Linear(256, num_classes))

        self.lateralfc1 = tdConv(nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, groups=1, bias=False))
        self.lateralfc2 = tdConv(nn.Conv1d(1, 1, kernel_size=5, stride=1, padding=2, groups=1, bias=False))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, newBatchNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def dropout_layer(self, x):
        d = torch.mean(x, dim=2)
        p = (d - d.min()) / (d.max() - d.min())
        p = 1 - (1 - p) * self.drop_rate / (1 - p.mean())
        p = torch.clamp(p, min=0., max=1.)
        d = torch.bernoulli(p)
        d = d.div(torch.where(p > 0, p, torch.tensor(1.).to(x.device)))
        d = torch.stack((d,) * steps, 2).to(x.device)
        return x * d

    def forward(self, x: Tensor) -> Tensor:
        features = self.features(x)
        out = self.spike1(features)
        out = self.avg(out)
        out = out.view(out.shape[0], -1, out.shape[4])

        if self.training:
            out = torch.cat((out, out), 0)
        out = self.classifier1(out)
        out = self.outspike1(out, self.lateralfc1(out.unsqueeze(1)).squeeze(1))
        if self.drop_rate > 0 and self.training:
            out = self.dropout_layer(out)
        out = self.classifier2(out)
        out = self.outspike2(out, self.lateralfc2(out.unsqueeze(1)).squeeze(1))
        out = torch.sum(out, dim=2) / steps
        if self.training:
            return out[:len(out) // 2], out[len(out) // 2:]
        else:
            return out


def _load_state_dict(model: nn.Module, model_url: str, progress: bool) -> None:
    pattern = re.compile(
        r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')

    state_dict = load_state_dict_from_url(model_url, progress=progress)
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    model.load_state_dict(state_dict)


def _densenet(
    args,
    arch: str,
    growth_rate: int,
    block_config: Tuple[int, int, int],
    num_init_features: int,
    pretrained: bool,
    progress: bool,
    **kwargs: Any
) -> DenseNet:
    model = DenseNet(args, growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, model_urls[arch], progress)
    return model


def densenet102(args=None, pretrained: bool = False, progress: bool = True, **kwargs: Any) -> DenseNet:
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
        memory_efficient (bool) - If True, uses checkpointing. Much more memory efficient,
          but slower. Default: *False*. See `"paper" <https://arxiv.org/pdf/1707.06990.pdf>`_.
    """
    return _densenet(args, 'densenet102', 32, (12, 12, 24), 256, pretrained, progress, **kwargs)