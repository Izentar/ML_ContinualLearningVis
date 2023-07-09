import torch
from  model.statistics.base import ModuleStatData, get_hash
from collections.abc import Sequence
import json
from utils import pretty_print as pp

class LayerBase():
    def __init__(self, scale_file:dict|str=None, scale:float=1.) -> None:
        self.new_cl = False
        self.loss_dict = {}
        self.scales = {}
        self.file_name = None
        self.from_dict = None
        if(isinstance(scale_file, str)):
            self.scales = json.load(open(scale_file))
            self.default_scale = scale
            self.file_name = scale_file
        elif(isinstance(scale_file, dict)):
            self.scales = scale_file
            self.default_scale = scale
            self.from_dict = True
        else:
            self.default_scale = scale

        self._print_scale_file()


    def _print_scale_file(self):
        s = f"{pp.COLOR.NORMAL}INFO: For layerloss {type(self).__name__} used "
        if(self.file_name is not None):
            s += f"file {self.file_name} "
        elif(self.from_dict is not None):
            s += f"dict "

        s += f"with default value: {pp.COLOR.NORMAL_2}'{self.default_scale}'{pp.COLOR.NORMAL} "
        if(len(self.scales) != 0):
            s += f"and with values:\n{pp.COLOR.NORMAL_2}{self.scales}"
            
        pp.sprint(s)

    def set_current_class(self, cl:torch.Tensor|int|Sequence):
        if(isinstance(cl, torch.TensorType)):
            self.current_cl = cl.item()
        else:
            self.current_cl = cl
        self.new_cl = True

    def _register_loss(self, full_name, output, value):
        name = (full_name, output.shape[1:])
        if(name in self.loss_dict):
            raise Exception(f'Loss for "{name}" was not processed. Tried to override loss.')
        self.loss_dict[name] = value * self.scales[full_name]

    def _gather_loss(self, loss) -> torch.Tensor:
        if(len(self.loss_dict) == 0):
            raise Exception("Loss dict is empty. Maybe tried to hook to the nonexistent layer?")
        sum_loss = torch.sum(torch.stack(list(self.loss_dict.values())))
        self.loss_dict = {}
        return loss + sum_loss
    
    def register_scale(self, full_name):
        # backup if no particular scale is registered
        if(self.scales.get(full_name) is None):
            self.scales[full_name] = self.default_scale

class DeepInversionTarget(LayerBase):
    def __init__(self, scale, multitarget=False, **kwargs) -> None:
        super().__init__(**kwargs)
        self.scale = scale   
        self.multitarget = multitarget

    def hook_fun(self, module:torch.nn.Module, full_name:str, layer_stat_data):
        def inner_single_target(module:torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
            data:ModuleStatData = layer_stat_data
            if(self.new_cl):
                data.lazy_flush()
                self.new_cl = False
            h = get_hash(k=self.current_cl, v=output.shape[1:])

            layer_mean_channel = data.mean_channel[h].requires_grad_(False).to(output.device)
            layer_var_channel = data.var_channel[h].requires_grad_(False).to(output.device)
            #output = output.view(output.shape[0], -1)

            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().view([input[0].shape[1], -1]).var(1, correction=0)

            val = torch.norm(layer_var_channel - var, 2) + torch.norm(layer_mean_channel - mean, 2)
            self._register_loss(full_name, output, self.scale * val)

        def inner_multi_target(module:torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
            data:ModuleStatData = layer_stat_data
            if(self.new_cl):
                data.lazy_flush()
                self.new_cl = False

            loss = []
            for idx, cl in enumerate(self.current_cl):
                h = get_hash(k=cl, v=output.shape[1:])
                current_input = input[0][idx]

                layer_mean_channel = data.mean_channel[h].requires_grad_(False).to(output.device)
                layer_var_channel = data.var_channel[h].requires_grad_(False).to(output.device)
                #new_output = output.view(output.shape[0], -1)

                mean = current_input.mean([1, 2])
                var = current_input.view([current_input.shape[0], -1]).var(1, correction=0)
                val = torch.norm(layer_var_channel - var, 2) + torch.norm(layer_mean_channel - mean, 2)
                loss.append(val)

            loss = torch.sum(torch.stack(loss))
            self._register_loss(full_name, output, self.scale * loss)

        self.register_scale(full_name)
        if(self.multitarget):
            return module.register_forward_hook(inner_multi_target)
        return module.register_forward_hook(inner_single_target)
        
    def gather_loss(self, loss) -> torch.Tensor:
        return self._gather_loss(loss)

class MeanNorm(LayerBase):
    def __init__(self, device, del_cov_after=False, scale=0.01, **kwargs) -> None:
        super().__init__(**kwargs)

        self.device = device
        self.scale = scale
        self.del_cov_after = del_cov_after
        print(f'LAYERLOSS::MEAN_NORM: Scaling {self.scale}')    
    
    def hook_fun(self, module:torch.nn.Module, full_name:str, layer_stat_data):
        def inner(module:torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
            data:ModuleStatData = layer_stat_data
            if(self.new_cl):
                data.lazy_flush()
                self.new_cl = False
            h = get_hash(k=self.current_cl, v=output.shape[1:])
            mean = data.mean[h].requires_grad_(False).to(self.device)
            cov_inverse = data.cov_inverse(del_cov_after=self.del_cov_after)[h].requires_grad_(False).to(self.device)
            output = output.view(output.shape[0], -1).to(self.device)

            mean_diff = output - mean
            val = self.scale * torch.sum(
                torch.diag(torch.linalg.multi_dot((mean_diff, cov_inverse, mean_diff.T)))
            ).to(input[0].device)
            self._register_loss(full_name, output, val)

        self.register_scale(full_name)
        return module.register_forward_hook(inner)
        
    def gather_loss(self, loss) -> torch.Tensor:
        return self._gather_loss(loss)

class DeepInversionFeatureLoss(LayerBase):
    '''
        Implementation of the forward hook to track feature statistics and compute a loss on them.
        Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)

    def hook_fun(self, module:torch.nn.Module, name:str, tree_name:str, full_name:str):
        def inner(module:torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
            # hook co compute deepinversion's feature distribution regularization
            nch = input[0].shape[1]

            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

            # forcing mean and variance to match between two distributions
            # other ways might work better, e.g. KL divergence
            r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
                module.running_mean.data.type(var.type()) - mean, 2)

            self._register_loss(full_name, output, r_feature)
            # must have no output
        self.register_scale(full_name)
        return module.register_forward_hook(inner)

    def gather_loss(self, loss) -> torch.Tensor:
        return self._gather_loss(loss)

class LayerGradPruning(LayerBase):
    def __init__(self, device, percent, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.device = device
        self.percent = percent
        self.by_class  = dict()

    def hook_fun(self, module:torch.nn.Module, full_name:str, layer_stat_data):           
        def inner(module:torch.nn.Module, grad_input:torch.Tensor, grad_output:torch.Tensor):
            """
                Sort std by descending order and zeros neurons that have the biggest std in given layer.
            """
            data:ModuleStatData = layer_stat_data
            if(self.new_cl):
                data.lazy_flush()
                self.new_cl = False
            h = get_hash(k=self.current_cl, v=grad_output[0].shape[1:])
            std = data.std[h].to(self.device)
            to = int(std.shape[0] * self.percent)
            indices = torch.argsort(std, descending=True)[:to]
            grad_input = grad_input[0].clone()
            grad_input_view = grad_input.view(grad_input.shape[0], -1)
            grad_input_view[:, indices] = 0.0
            # return val should match input
            # Since backprop works in reverse, grad_output is what got propagated 
            # from the next layer while grad_input is what will be sent to the previous one.
            return (grad_input, )
        
        self.register_scale(full_name)
        return module.register_full_backward_hook(inner)

class LayerGradActivationPruning(LayerBase):
    def __init__(self, percent, **kwargs) -> None:
        super().__init__(**kwargs)
        
        self.percent = percent
        self.by_class  = dict()

    def hook_fun(self, module:torch.nn.Module, name:str, tree_name, full_name):
        def inner(module:torch.nn.Module, grad_input:torch.Tensor, grad_output:torch.Tensor):
            """
                Look which neurons have the smallest gadient and zero them.
            """
            grad_input = grad_input[0].clone()
            grad_input_view = grad_input.view(grad_input.shape[0], -1)
            B = grad_input_view.shape[0]
            k = int(grad_input_view.shape[1] * self.percent)
            # magic from https://discuss.pytorch.org/t/is-it-possible-use-torch-argsort-output-to-index-a-tensor/134327/2
            i0 = torch.arange(B).unsqueeze(-1).expand(B, k)
            i1 = torch.topk(grad_input_view, k, dim = 1, largest=False).indices.expand(B, k)
            i1_1 = torch.topk(grad_input_view, k, dim = 1, largest=True).indices.expand(B, k)
            grad_input_view[i0, i1] = 0.0
            grad_input_view[i0, i1_1] = 0.0
            return (grad_input, )

        self.register_scale(full_name)
        return module.register_full_backward_hook(inner)
