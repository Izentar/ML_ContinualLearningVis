import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.nn.functional import cross_entropy
from torch.autograd import Variable


from model import base

import sys
import numpy as np
import math

import os, sys
from utils.data_manipulation import count_parameters

##############################################################################
# Based on source https://github.com/bearpaw/pytorch-classification/
##############################################################################

##############################################################################
# VGG
##############################################################################

'''VGG for CIFAR10. FC layers are removed.
(c) YANG, Wei 
'''

class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Linear(512, num_classes)
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def vgg11(**kwargs):
    """VGG 11-layer model (configuration "A")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['A']), **kwargs)
    return model

def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    model = VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)
    return model

def vgg13(**kwargs):
    """VGG 13-layer model (configuration "B")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['B']), **kwargs)
    return model

def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    model = VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)
    return model

def vgg16(**kwargs):
    """VGG 16-layer model (configuration "D")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['D']), **kwargs)
    return model

def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    model = VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)
    return model

def vgg19(**kwargs):
    """VGG 19-layer model (configuration "E")
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = VGG(make_layers(cfg['E']), **kwargs)
    return model

def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    model = VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)
    return model

class VGGBaseModel(base.CLBase):
    def __init__(self, loss_f=None, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_f = loss_f if loss_f is not None else cross_entropy

    def training_step_normal(self, batch):
        x, y = batch
        y_model_out = self(x)
        loss_classification = self.loss_f(y_model_out, y)
        self.log("train_loss/classification", loss_classification)
        self.train_acc(y_model_out, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss_classification

    def training_step_dream(self, batch):
        x, y = batch
        y_model_out = self(x)
        loss_classification = self.loss_f(y_model_out, y)
        self.log("train_loss_dream/classification", loss_classification)
        self.train_acc_dream(y_model_out, y)
        self.log("train_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)
        return loss_classification

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_model_out = self(x)
        val_loss = self.loss_f(y_model_out, y)
        self.log("val_loss", val_loss)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(y_model_out, y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_model_out = self(x)
        test_loss = self.loss_f(y_model_out, y)
        self.log("test_loss", test_loss)
        self.test_acc(y_model_out, y)
        self.log("test_acc", self.test_acc)

    def get_objective_target(self):
        return "classifier"

class VGGDefault(VGGBaseModel):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)
        self.vgg = None # where to store model

    def forward(self, *args):
        return self.vgg(*args)

    def get_objective_target(self):
        ret = super().get_objective_target()
        return "vgg_" + ret

class VGG11(VGGDefault):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.vgg = vgg11(num_classes=num_classes)

class VGG11_BN(VGGDefault):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.vgg = vgg11_bn(num_classes=num_classes)

class VGG13(VGGDefault):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.vgg = vgg13(num_classes=num_classes)

class VGG13_BN(VGGDefault):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.vgg = vgg13_bn(num_classes=num_classes)

class VGG16(VGGDefault):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.vgg = vgg16(num_classes=num_classes)

class VGG16_BN(VGGDefault):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.vgg = vgg16_bn(num_classes=num_classes)

class VGG19(VGGDefault):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.vgg = vgg19(num_classes=num_classes)

class VGG19_BN(VGGDefault):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.vgg = vgg19_bn(num_classes=num_classes)