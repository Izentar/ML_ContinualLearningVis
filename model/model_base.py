from abc import abstractmethod

class ModelBase():
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