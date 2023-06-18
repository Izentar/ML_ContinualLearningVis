
from torchvision.models import resnet18, resnet, ResNet18_Weights, resnet34, ResNet34_Weights
from torchvision import models
from torch import nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from model.model.base import ModelBase

class ResNetBase(nn.Module, ModelBase):
    def _initialize_weights(self):
        super()._initialize_weights()
        if hasattr(self, 'zero_init_residual') and self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, resnet.Bottleneck) and m.bn3.weight is not None:
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, resnet.BasicBlock) and m.bn2.weight is not None:
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

class ResNet18(ResNetBase):
    def __init__(self, num_classes, default_weights:bool=None, *args, **kwargs):
        super().__init__()

        if(default_weights):
            default_weights = ResNet18_Weights.DEFAULT
        self.resnet = resnet18(weights=default_weights)
        # reinit layer for finetuning
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self._initialize_weights()

    def forward(self, x, **kwargs):
        return self.resnet(x)

    def get_objective_layer_name(self):
        return "resnet.fc"

    def get_root_name(self):
        return "resnet"

    def get_objective_layer(self):
        return self.resnet.fc

    def get_objective_layer_output_shape(self):
        return (self.resnet.fc.out_features,)
    
    @property
    def name(self):
        return "ResNet18"
        
class ResNet34(ResNetBase):
    def __init__(self, num_classes, default_weights:bool=None, *args, **kwargs):
        super().__init__()

        if(default_weights):
            default_weights = ResNet34_Weights.DEFAULT
        self.resnet = resnet34(weights=default_weights)
        # reinit layer for finetuning
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self._initialize_weights()

    def forward(self, x, **kwargs):
        return self.resnet(x)

    def get_objective_layer_name(self):
        return "resnet.fc"

    def get_root_name(self):
        return "resnet"

    def get_objective_layer(self):
        return self.resnet.fc

    def get_objective_layer_output_shape(self):
        return (self.resnet.fc.out_features,)

    @property
    def name(self):
        return "ResNet34"

class BasicBlock(nn.Module):
    expansion = 1
 
    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out
 
 
class Bottleneck(nn.Module):
    expansion = 4
 
    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)
 
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
 
    def forward(self, x):
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = nn.functional.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = nn.functional.relu(out)
        return out
 
 
class CreatorResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(CreatorResNet, self).__init__()
        self.in_planes = 64
 
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
 
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
 
    def forward(self, x):
        x = self.conv1(x)

        x = self.bn1(x)
        out = nn.functional.relu(x)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = nn.functional.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        return out
 
class CustomResNet34(nn.Module, ModelBase):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()

        self.resnet = CreatorResNet(BasicBlock, [3,4,6,3], num_classes)

    def forward(self, x, **kwargs):
        return self.resnet(x)

    def get_objective_layer_name(self):
        return "resnet.linear"

    def get_root_name(self):
        return "resnet"

    def get_objective_layer(self):
        return self.resnet.linear

    def get_objective_layer_output_shape(self):
        return (self.resnet.linear.out_features,)
    
    @property
    def name(self):
        return "CustomResNet34"

class Resnet20C100(ResNetBase):
    def __init__(self, pretrained=True, *args, **kwargs):
        super().__init__()

        self.resnet = ptcv_get_model("resnet20_cifar100", pretrained=pretrained)
        self._initialize_weights()

    def forward(self, x, **kwargs):
        return self.resnet(x)

    def get_objective_layer_name(self):
        return "resnet.output"

    def get_root_name(self):
        return "resnet"

    def get_objective_layer(self):
        return self.resnet.output

    def get_objective_layer_output_shape(self):
        return (self.resnet.output.out_features, )
    
    @property
    def name(self):
        return "Resnet20C100"