
from torchvision.models import resnet18
from torch import nn
from pytorchcv.model_provider import get_model as ptcv_get_model

class ResNet18(nn.Module):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()

        self.resnet = resnet18(num_classes=num_classes)

    def forward(self, x, **kwargs):
        return self.resnet(x)

    def get_objective_layer_name(self):
        return "resnet_fc"

    def get_root_name(self):
        return "resnet_"

class Resnet20C100(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        self.resnet = ptcv_get_model("resnet20_cifar100", pretrained=pretrained)

    def forward(self, x, **kwargs):
        return self.resnet(x)

    def get_objective_layer_name(self):
        return "resnet_output"

    def get_root_name(self):
        return "resnet_"

    