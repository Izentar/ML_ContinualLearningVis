
from torchvision.models import resnet18, resnet
from torchvision import models
from torch import nn
from pytorchcv.model_provider import get_model as ptcv_get_model
from model.model_base import ModelBase

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
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__()

        self.resnet = resnet18(num_classes=num_classes)
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