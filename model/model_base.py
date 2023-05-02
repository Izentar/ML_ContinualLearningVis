from abc import abstractmethod
import math
from torch import nn

class ModelBase():
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    @abstractmethod
    def forward(self, x, **kwargs):
        pass

    @abstractmethod
    def get_objective_layer_name(self):
        pass

    @abstractmethod
    def get_root_name(self):
        """
            The name of the variable where the model is if using model from other framework.
            May return "" (empty string) if the model is implemented inside the class.
        """
        pass

    @abstractmethod
    def get_objective_layer(self):
        """
            Return objective layer where the output can be used in loss function.
        """
        pass

    @abstractmethod
    def get_objective_layer_output_shape(self):
        """
            Return objective layer shape where the output can be used in loss function.
            The shape should be a tuple that does not contain batch size.
        """
        pass

    @abstractmethod
    def init_weights(self):
        pass