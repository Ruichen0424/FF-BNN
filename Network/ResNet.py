import torch
import torch.nn as nn
from .BNN import *


def conv3x3(in_planes, out_planes, stride=1, padding=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False)

def Bconv3x3(in_planes, out_planes, stride=1, padding=0):
    """3x3 convolution without padding"""
    return BinaryConv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=padding, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        
        self.padding_layer = nn.ConstantPad2d(padding=1, value=-1)
        self.conv1 = Bconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        
        self.downsample = downsample

    def forward(self, x):
        
        identity = x
        
        out = self.padding_layer(x)
        out = self.conv1(out)
        out = self.bn1(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, imagenet=True, num_classes=1000):
        super().__init__()

        self.inplanes = 64
        self.imagenet = imagenet
        
        if imagenet:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.conv1 = conv3x3(3, self.inplanes)
            self.maxpool = nn.Identity()
            
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        self.sn1 = nn.ReLU()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.AvgPool2d(kernel_size=2, stride=stride),
                conv1x1(self.inplanes, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.sn1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

def ResNet18(num_classes=1000, imagenet=True):
    model = ResNet(BasicBlock, [4, 4, 4, 4], imagenet=imagenet, num_classes=num_classes)
    return model

def ResNet34(num_classes=1000, imagenet=True):
    model = ResNet(BasicBlock, [6, 8, 12, 6], imagenet=imagenet, num_classes=num_classes)
    return model