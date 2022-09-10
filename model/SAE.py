from model import base
import torch
from torch import nn, sigmoid
from torch.nn.functional import relu, cross_entropy, mse_loss
from torch.autograd.variable import Variable
from robustness.tools import custom_modules
import math

class SAE_CIFAR(nn.Module):
    def __init__(self, num_classes, hidd1=256, last_hidd_layer=32):
        super().__init__()
        self.hidd1 = hidd1
        self.last_hidd_layer = last_hidd_layer
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
        )

        self.fc1_1 = nn.Linear(in_features=50176, out_features=hidd1)
        self.fc1_2 = nn.Linear(in_features=hidd1, out_features=last_hidd_layer)

        #self.fc3_2 = nn.Linear(in_features=last_hidd_layer, out_features=hidd1)
        #self.fc2_1 = nn.Linear(in_features=hidd1, out_features=50176)
        #self.conv3 = nn.ConvTranspose2d(
        #    in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
        #)
        #self.conv4 = nn.ConvTranspose2d(
        #    in_channels=64, out_channels=3, kernel_size=(3, 3), stride=(1, 1)
        #)

        #self.fake_relu = custom_modules.FakeReLUM()
        #self.fc = nn.Linear(in_features=last_hidd_layer, out_features=num_classes)

        self._initialize_weights()
    
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

    # based on https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures
    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        xe_latent_pre_relu = self.forward_encoder(
            x, 
            with_latent=with_latent, 
            fake_relu=fake_relu, 
            no_relu=no_relu
        )

        #x_hat = self.forward_decoder(
        #    xe_latent, 
        #    shp,
        #    with_latent=with_latent, 
        #    fake_relu=fake_relu, 
        #    no_relu=no_relu
        #)

        #if with_latent:
        #    if no_relu:
        #        return (encoder_hat, xe_latent_pre_relu), x_hat
        #    else:
        #        return encoder_hat, xe_latent_second
        #return encoder_hat, xe_latent_pre_relu, x_hat

        return xe_latent_pre_relu
        
    def forward_encoder(self, x, with_latent=False, fake_relu=False, no_relu=False):
        xe = relu(self.conv1(x))
        xe = relu(self.conv2(xe))
        shp = [xe.shape[0], xe.shape[1], xe.shape[2], xe.shape[3]] # 32 batch | 64 channels | 28 x | 28 y

        xe = xe.reshape(-1, shp[1] * shp[2] * shp[3])
        xe = relu(self.fc1_1(xe))
        xe_latent_pre_relu = self.fc1_2(xe)

        #xe_latent_pre_relu = self.fc2_3(xe)
        #xe_latent = relu(xe_latent_pre_relu)
        #xe_latent_second = self.fake_relu(xe_latent_pre_relu) if fake_relu else xe_latent
        #encoder_hat = self.fc(xe_latent_second)
        return xe_latent_pre_relu#, xe_latent, xe_latent_second, encoder_hat, shp

    def forward_decoder(self, xe, shp, with_latent=False, fake_relu=False, no_relu=False):
        xd = relu(self.fc1_2(xe))
        xd = relu(self.fc2_1(xd))
        xd = torch.reshape(xd, (shp[0], shp[1], shp[2], shp[3]))
        xd = relu(self.conv3(xd))
        # xd = F.upsample(xd,30)
        return sigmoid(self.conv4(xd))

    def get_objective_layer_name(self):
        return "fc1_2"

class SAE_CIFAR_TEST(SAE_CIFAR):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.linear_cl = nn.Linear(in_features=self.last_hidd_layer, out_features=num_classes)
        self._initialize_weights()

    def forward(self, x, with_latent=False, fake_relu=False, no_relu=False):
        xe = super().forward(x, with_latent=with_latent, fake_relu=fake_relu, no_relu=no_relu)
        xe = relu(self.linear_cl(xe))
        return xe

    def get_test_value(self):
        return self.linear_cl.weight[0][0]

class SAE_standalone(base.CLBase):
    def __init__(self, num_classes, loss_f=None, reconstruction_loss_f=None, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.model = SAE_CIFAR(num_classes)

        self.loss_f = loss_f if loss_f is not None else cross_entropy
        self.reconstruction_loss_f = reconstruction_loss_f if reconstruction_loss_f is not None else mse_loss

    def training_step_normal(self, batch):
        x, y = batch
        y_hat, y_reconstruction = self(x)
        loss_classification = self.loss_f(y_hat, y)
        loss_reconstruction = self.reconstruction_loss_f(y_reconstruction, Variable(x))
        alpha = 0.05
        loss = alpha * loss_classification + (1 - alpha) * loss_reconstruction
        self.log("train_loss/total", loss)
        self.log("train_loss/classification", loss_classification)
        self.log("train_loss/reconstrucion", loss_reconstruction)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def training_step_dream(self, batch):
        x, y = batch
        y_hat, y_reconstruction = self(x)
        loss_classification = self.loss_f(y_hat, y)
        loss_reconstruction = self.reconstruction_loss_f(y_reconstruction, Variable(x))
        alpha = 0.05
        loss = alpha * loss_classification + (1 - alpha) * loss_reconstruction
        self.log("train_loss_dream/total", loss)
        self.log("train_loss_dream/classification", loss_classification)
        self.log("train_loss_dream/reconstrucion", loss_reconstruction)
        self.train_acc_dream(y_hat, y)
        self.log("train_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat, _ = self(x)
        val_loss = self.loss_f(y_hat, y)
        self.log("val_loss", val_loss)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(y_hat, y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        test_loss = self.loss_f(y_hat, y)
        self.log("test_loss", test_loss)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc)

    def get_objective_target(self):
        return "model_fc"

    def forward(self, *args):
        return self.model(*args)

