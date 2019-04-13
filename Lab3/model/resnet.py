import torch.nn as nn

def conv1x1(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=1, groups=1, bias=False)

def conv3x3(in_channel, out_channel):
    return nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1, groups=1, bias=False)

class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channel, out_channel, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

        self.downsample = downsample 

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x) 

        out += identity 
        out = self.relu2(out)

        return out

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channel, out_channel, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(in_channel, out_channel)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu1 = nn.ReLU()

        self.conv2 = conv3x3(out_channel, out_channel)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu2 = nn.ReLU()

        self.conv3 = conv1x1(out_channel, out_channel*self.expansion)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu3 = nn.ReLU()

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

        if self.downsample is not None:
            identity = self.downsample(x) 

        out += identity
        out = self.relu2(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000, width_per_group=64):
        super(ResNet, self).__init__()
        in_planes = [width_per_group * 2 ** i for i in range(4)]
        self.in_plane = in_planes[0]

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, layers[0], in_planes[0])
        self.layer2 = self._make_layer(block, layers[1], in_planes[1])
        self.layer3 = self._make_layer(block, layers[2], in_planes[2])
        self.layer4 = self._make_layer(block, layers[3], in_planes[3])

        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_planes[-1] * block.expansion, num_classes)

    def _make_layer(self, block, block_num, planes):
        layers = []
        downsample = None
        if self.in_plane != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_plane, planes * block.expansion),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers.append(block(self.in_plane, planes, downsample=downsample))
        self.in_plane = planes * block.expansion
        for _ in range(1, block_num):
            layers.append(block(self.in_plane, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.shape[0], -1)
        y = self.fc(x)

        return y 

def resnet18(num_classes=1000):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

def resnet34(num_classes=1000):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes)

def resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)

def resnet101(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes)

def resnet152(num_classes=1000):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes)
