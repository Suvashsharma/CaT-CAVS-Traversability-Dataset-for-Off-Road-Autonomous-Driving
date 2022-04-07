import torch
import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

##        if self.downsample is not None:
##            print('Downsample:', self.downsample)

        

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)


        out = self.conv2(out)
        out = self.bn2(out)

##        print('Size of out:', out.size())


        if self.downsample is not None:
##            print('Should be downsampled.')
##            print('Before downsample:', residual.size())
            residual = self.downsample(x)
##            print('After downsample:',residual.size())
##            print('Downsample:', self.downsample)
##            print('Size of input:', x.size())
##            print('Output size:', out.size())
##            print('Residual_x downsampled to:', residual.size())
            


        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

##        print(self.downsample)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, deep_base=True):
        super(ResNet, self).__init__()
        self.deep_base = False
        if not self.deep_base:
            self.inplanes = 64
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = nn.BatchNorm2d(64)
        else:
##            print("INVOKED IN ELSE")
            self.inplanes = 128
            self.conv1 = conv3x3(3, 64, stride=2)
            self.bn1 = nn.BatchNorm2d(64)
            self.conv2 = conv3x3(64, 64)
            self.bn2 = nn.BatchNorm2d(64)
            self.conv3 = conv3x3(64, 128)
            self.bn3 = nn.BatchNorm2d(128)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
##        print("Layer1:", self.layer1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
##        print("Layer2:", self.layer2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
##        print("Layer3:", self.layer3)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
##        print("Layer4:", self.layer4)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

##        for m in self.modules():
##            if isinstance(m, nn.Conv2d):
##                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
##            elif isinstance(m, nn.BatchNorm2d):
##                nn.init.constant_(m.weight, 1)
##                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
##        print("Strides Inside _make_layer:", stride)
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
##            print("Stride = %d. So downsample is added." %stride)
##        else:
##            print("Stride = %d. So no downsample is added.+" %stride)

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
##            print("Number of blocks:", blocks)
            layers.append(block(self.inplanes, planes))
##        print("Downsample:", downsample)

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        if self.deep_base:
            x = self.relu(self.bn2(self.conv2(x)))
            x = self.relu(self.bn3(self.conv3(x)))
        x = self.maxpool(x)
        
##        print("layer2:", self.layer2)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet18(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
##        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
        model_path = '/home/ece_tech_5323/Suvash/detection_and_segmentation/psp_net/model/trained_models/resnet18-5c106cde.pth'
        model.load_state_dict(torch.load(model_path))
        print("ResNet-18 Loaded")
##        model = torch.nn.DataParallel(model, device_ids = range(torch.cuda.device_count()))
    return model


def resnet34(pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
##        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
        model_path = '/home/ece_tech_5323/Suvash/detection_and_segmentation/psp_net/model/trained_models/resnet34-333f7ec4.pth' 
        model.load_state_dict(torch.load(model_path))
        print("ResNet-34 Loaded")
        
    return model


def resnet50(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
        model_path ='/home/ece_tech_5323/Suvash/detection_and_segmentation/psp_net/model/trained_models/resnet50-19c8e357.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
        print("ResNet-50 Loaded")
    return model


def resnet101(pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
        model_path = '/home/ece_tech_5323/Suvash/detection_and_segmentation/psp_net/model/trained_models/resnet101-5d3b4d8f.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
        print("ResNet-101 Loaded")
    return model


def resnet152(pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        # model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
        model_path = '/home/ece_tech_5323/Suvash/detection_and_segmentation/psp_net/model/trained_models/resnet152-b121ed2d.pth'
        model.load_state_dict(torch.load(model_path), strict=False)
        print("ResNet-152 Loaded")
    return model
