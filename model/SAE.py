from model import base
import torch
from torch import nn, sigmoid
from torch.nn.functional import relu, cross_entropy, mse_loss
from torch.autograd.variable import Variable
import math
from model.model_base import ModelBase
from model.activation_layer import gaussA, conjunction, GaussA

class SAE_CIFAR(nn.Module, ModelBase):
    def __init__(self, num_classes, ln_hidden1=256, with_reconstruction=True):
        super().__init__()
        self.with_reconstruction = with_reconstruction
        self.ln_hidden1 = ln_hidden1
        self.conv_enc1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv_enc2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
        )

        self.relu = torch.nn.ReLU(inplace=True)

        self.ln_enc1 = nn.Linear(in_features=50176, out_features=ln_hidden1)
        self.ln_encode_cl = nn.Linear(in_features=ln_hidden1, out_features=num_classes)

        self.ln_decode_cl = nn.Linear(in_features=num_classes, out_features=ln_hidden1)
        self.ln_dec1 = nn.Linear(in_features=ln_hidden1, out_features=50176)
        self.conv_dec1 = nn.ConvTranspose2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv_dec2 = nn.ConvTranspose2d(
            in_channels=32, out_channels=3, kernel_size=(3, 3), stride=(1, 1)
        )

        #self.fake_relu = custom_modules.FakeReLUM()
        #self.fc = nn.Linear(in_features=last_hidd_layer, out_features=num_classes)

        self._initialize_weights()

        self.custom_forward_decoder_f = lambda xe_latent_pre_relu, shp: xe_latent_pre_relu
        if (self.with_reconstruction):
            self.custom_forward_decoder_f = self._decode_part
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
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

    def init_weights(self):
        self._initialize_weights()

    def _decode_part(self, xe_latent_pre_relu, linear_to_conv_shape):
        x = self.forward_decoder_class(xe_latent_pre_relu)

        image_reconstruction = self.forward_decoder_hidden(x, linear_to_conv_shape)

        return xe_latent_pre_relu, {
            'image_reconstruction': image_reconstruction,
        }

    def forward_encoder(self, x):
        xe, conv_to_linear_shape = self.forward_encoder_hidden(
            x, 
        )
        #xe = relu(self.ln_enc2(xe))
        xe_latent_pre_relu = self.forward_encoder_class(xe)
        return xe_latent_pre_relu, conv_to_linear_shape

    # based on https://robustness.readthedocs.io/en/latest/example_usage/training_lib_part_2.html#training-with-custom-architectures
    def forward(self, x, **kwargs):
        xe_latent_pre_relu, conv_to_linear_shape = self.forward_encoder(x)

        return self.custom_forward_decoder_f(xe_latent_pre_relu, conv_to_linear_shape)
        
    def forward_encoder_class(self, x):
        return self.ln_encode_cl(x)

    def forward_encoder_hidden(self, x):
        xe = self.relu(self.conv_enc1(x))
        xe = self.relu(self.conv_enc2(xe))
        conv_to_linear_shape = [xe.shape[0], xe.shape[1], xe.shape[2], xe.shape[3]] # 32 batch | 64 channels | 28 x | 28 y

        xe = xe.reshape(-1, conv_to_linear_shape[1] * conv_to_linear_shape[2] * conv_to_linear_shape[3])
        xe = self.relu(self.ln_enc1(xe))

        #xe_latent_pre_relu = self.fc2_3(xe)
        #xe_latent = relu(xe_latent_pre_relu)
        #xe_latent_second = self.fake_relu(xe_latent_pre_relu) if fake_relu else xe_latent
        #encoder_hat = self.fc(xe_latent_second)
        return xe, conv_to_linear_shape#, xe_latent, xe_latent_second, encoder_hat, conv_to_linear_shape

    def forward_decoder_class(self, xe):
        xd = self.relu(self.ln_decode_cl(xe))
        return xd

    def forward_decoder_hidden(self, xe, linear_to_conv_shape):
        xd = self.relu(self.ln_dec1(xe))
        xd = torch.reshape(xd, (linear_to_conv_shape[0], linear_to_conv_shape[1], linear_to_conv_shape[2], linear_to_conv_shape[3]))
        xd = self.relu(self.conv_dec1(xd))
        # xd = F.upsample(xd,30)
        return sigmoid(self.conv_dec2(xd))

    def get_objective_layer_name(self):
        return "ln_encode_cl"

    def get_root_name(self):
        return ""

    def get_objective_layer(self):
        return self.ln_encode_cl

    def get_objective_layer_output_shape(self):
        return (self.ln_encode_cl.out_features, )

class SAE_CIFAR_TEST(SAE_CIFAR):
    def __init__(self, num_classes, new_hidd_layer, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.ln_enc1 = nn.Linear(in_features=50176, out_features=new_hidd_layer)
        self.ln_enc2 = nn.Linear(in_features=new_hidd_layer, out_features=self.ln_hidden1)
        self.ln_encode_cl = nn.Linear(in_features=self.ln_hidden1, out_features=num_classes)
        self._initialize_weights()

    def forward_encoder(self, x):
        xe, conv_to_linear_shape = self.forward_encoder_hidden(
            x, 
        )
        xe = self.relu(self.ln_enc2(xe))
        xe_latent_pre_relu = self.forward_encoder_class(xe)
        return xe_latent_pre_relu, conv_to_linear_shape

class SAE_CIFAR_GAUSS(SAE_CIFAR):
    def __init__(self, num_classes, ln_hidden1=256, with_reconstruction=True):
        super().__init__(num_classes=num_classes, ln_hidden1=ln_hidden1, with_reconstruction=with_reconstruction)

        self.gauss_cov = GaussA(0.1)
        self.gauss_linear = GaussA(30)

    def forward(self, x, **kwargs):
        xe = self.gauss_cov(self.conv_enc1(x))
        #print(torch.mean(torch.abs(xe), dim=(1, 2, 3)))
        xe = self.gauss_cov(self.conv_enc2(xe))
        #print(torch.mean(torch.abs(xe), dim=(1, 2, 3)))
        shp = [xe.shape[0], xe.shape[1], xe.shape[2], xe.shape[3]] # 32 batch | 64 channels | 28 x | 28 y

        #a = torch.randn((3, 3))
        #print(a)
        #b = torch.exp(-0.5 * a*a)
        #print(b)
        #exit()

        xe = xe.reshape(-1, shp[1] * shp[2] * shp[3])
        r = self.ln_enc1(xe)
        #print(torch.sum(r).item(), torch.min(r).item(), torch.max(r).item(), torch.mean(r).item(), r[0][0].item())
        #tmp = r[0][0]
        #print((-0.001 * tmp * tmp).item())
        #print(torch.exp(-0.001 * tmp * tmp).item())
        #r = torch.sigmoid(r)
        #xe = relu(r)
        xe = self.gauss_linear(r)
        #print(torch.mean(xe))
        #print(torch.sum(xe).item(), torch.min(xe).item(), torch.max(xe).item(), torch.mean(xe).item(), xe[0][0].item())
        #print(xe.shape, torch.linalg.norm(xe, dim=1))
        #print()
        xe_latent_pre_relu = self.ln_encode_cl(xe)

        return xe_latent_pre_relu

    def forward_decoder(self, xe, shp):
        xd = gaussA(self.ln_decode_cl(xe))
        xd = gaussA(self.ln_dec1(xd))
        xd = torch.reshape(xd, (shp[0], shp[1], shp[2], shp[3]))
        xd = gaussA(self.conv_dec1(xd))
        return sigmoid(self.conv_dec2(xd))

class SAE_CIFAR_CONJ(SAE_CIFAR_GAUSS):
    def __init__(self, num_classes, ln_hidden1=256, with_reconstruction=True):
        super().__init__(num_classes=num_classes, ln_hidden1=ln_hidden1, with_reconstruction=with_reconstruction)

        self.ln_encode_cl = nn.Linear(in_features=math.ceil(ln_hidden1 / 2), out_features=num_classes)

    def forward_encoder(self, x):
        xe = gaussA(self.conv_enc1(x))
        xe = gaussA(self.conv_enc2(xe))
        shp = [xe.shape[0], xe.shape[1], xe.shape[2], xe.shape[3]] # 32 batch | 64 channels | 28 x | 28 y

        xe = xe.reshape(-1, shp[1] * shp[2] * shp[3])
        t = self.ln_enc1(xe)
        xe = conjunction(t)
        xe_latent_pre_relu = self.ln_encode_cl(xe)

        return xe_latent_pre_relu, shp