from typing import Callable

import torch
import torch.nn as nn
from torch.nn.functional import relu, avg_pool2d

from src.model.features_map import FeaturesMapModel
from src.utils.transformations import scale_by_norm


class DistLinear(nn.Module):
    """
    DistLinear layer class. This is a modification of the linear layer where before the dot product the input and the
    weights of the linear layer are scaled by their own norm. The result are then multiplied by a scale factor.
    """

    def __init__(self, size_in: int, size_out: int, scale_factor: int = 10):
        """
        Constructor of dist layer.
        @param size_in: Input features size.
        @param size_out: Output features size.
        @param scale_factor: Scale factor.
        """
        super(DistLinear, self).__init__()
        self.linear = torch.randn((size_in, size_out))
        self.scale_factor = scale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pytorch forward function.
        @param x: Input to predict.
        @return: Prediction of the input.
        """
        cos_dist = scale_by_norm(x) @ scale_by_norm(self.linear)
        scores = self.scale_factor * cos_dist
        return scores


def conv3x3(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """
    Function able to create a custom convolutional layer with kernel 3x3.
    @param in_planes: Input features.
    @param out_planes: Output features.
    @param stride: Stride.
    @return: Initialized layer.
    """
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False,
    )


class BasicBlock(nn.Module):
    """
    Basic block of ResNet.
    """
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        """
        ResNet basic block constructor.
        @param in_planes: Input features.
        @param planes: Hidden features.
        @param stride: Stride of first convolutional layer.
        """
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pytorch forward method.
        @param x: Input to predict.
        @return: Prediction of the input.
        """
        out = relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = relu(out)
        return out


class ResNet(nn.Module, FeaturesMapModel):
    """
    ResNet model taken by avalanche library.
    """

    def __init__(self, block: Callable, num_blocks: list[int], num_classes: int, nf: int, dist_linear: bool = False):
        """
        ResNet constructor.
        @param block: Function able to construct a ResNet block.
        @param num_blocks: Number of ResNet blocks for each layer.
        @param num_classes: Number of output classes.
        @param nf: Number of features in input of layers.
        @param dist_linear: Flag able to replace linear with dist linear in output layer.
        """
        super(ResNet, self).__init__()
        self.in_planes = nf
        self.conv1 = conv3x3(3, nf * 1)
        self.bn1 = nn.BatchNorm2d(nf * 1)
        self.layer1 = self._make_layer(block, nf * 1, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, nf * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, nf * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, nf * 8, num_blocks[3], stride=2)
        self.linear = (DistLinear if dist_linear else nn.Linear)(nf * 8 * block.expansion, num_classes)

    def _make_layer(self, block: Callable, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        """
        Method able to create a layer of ResNet.
        @param block: Function able to create a block.
        @param planes: Number of hidden features of the block.
        @param num_blocks: Number of blocks.
        @param stride: Stride applied to the first sublayer of the current layer.
        @return: Sequential of sublayers of the current layer.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def return_hidden(self, x: torch.Tensor) -> torch.Tensor:
        """
        Method able to return the latent representations of some input data.
        @param x: Input data.
        @return: Latent representations.
        """
        bsz = x.size(0)
        out = relu(self.bn1(self.conv1(x.view(bsz, 3, 32, 32))))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pytorch forward method.
        @param x: Input data.
        @return: Prediction data.
        """
        out = self.return_hidden(x)
        out = self.linear(out)
        return out


def ResNet18(n_classes, dist_linear=False):
    """
    Function able to create a ResNet18 model.
    @param n_classes: Number of classes in output layer.
    @param dist_linear: Flag that if is true replace linear layer with dist linear layer.
    @return: ResNet18 initialized model.
    """
    return ResNet(BasicBlock, [2, 2, 2, 2], n_classes, 20, dist_linear=dist_linear)
