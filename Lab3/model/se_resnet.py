import torch.nn as nn

from .se_module import SELayer
from .resnet import ResNet

def conv1x1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=1, bias=False)

def conv3x3(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, groups=1, bias=False)

class SEBasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channel, out_channel, reduction=16, downsample=None):
        super(SEBasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

        self.downsample = downsample 
        self.se = SELayer(out_channel, reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x) 

        out += identity 
        out = self.relu2(out)

        return out

class SEBottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channel, out_channel, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = conv1x1(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

        self.conv3 = conv3x3(out_channel, out_channel*self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu3 = nn.ReLU()
        
        self.se = SELayer(out_channel * self.expansion, reduction)

        self.downsample = downsample 

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.bn3(out)
        
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x) 

        out += identity
        out = self.relu2(out)

        return out

def se_resnet18(num_classes=1000):
    return ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def se_resnet34(num_classes=1000):
    return ResNet(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def se_resnet50(num_classes=1000):
    return ResNet(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes)

def se_resnet101(num_classes=1000):
    return ResNet(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes)

def se_resnet152(num_classes=1000):
    return ResNet(SEBottleneck, [3, 8, 36, 3], num_classes=num_classes)
