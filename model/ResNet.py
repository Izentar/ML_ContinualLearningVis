
from torchvision.models import resnet18
from torch import nn

class ResNet18(nn.Module):
    def __init__(self,  *args, **kwargs):
        super().__init__()

        self.resnet = resnet18(num_classes=10)

    def forward(self, x, **kwargs):
        return self.resnet(x)

    def get_objective_layer_name(self):
        return "resnet_fc"