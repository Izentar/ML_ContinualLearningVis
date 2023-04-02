
from torchvision.models import resnet18
from torch import nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from model.model_base import ModelBase

class ResNet18(nn.Module, ModelBase):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()

        self.resnet = resnet18(num_classes=num_classes)
        #replace_layer(self, 'self', torch.nn.ReLU, lambda a, b, x: torch.nn.LeakyReLU(negative_slope=0.05, inplace=x.inplace))

    def forward(self, x, **kwargs):
        return self.resnet(x)

    def get_objective_layer_name(self):
        return "resnet_fc"

    def get_root_name(self):
        return "resnet_"

    def get_objective_layer(self):
        return self.resnet.fc

    def get_objective_layer_output_shape(self):
        return (self.resnet.fc.out_features,)

class Resnet20C100(nn.Module, ModelBase):
    def __init__(self, pretrained=True, *args, **kwargs):
        super().__init__()

        self.resnet = ptcv_get_model("resnet20_cifar100", pretrained=pretrained)

    def forward(self, x, **kwargs):
        return self.resnet(x)

    def get_objective_layer_name(self):
        return "resnet_output"

    def get_root_name(self):
        return "resnet_"

    def get_objective_layer(self):
        return self.resnet.output

    def get_objective_layer_output_shape(self):
        return (self.resnet.output.out_features, )