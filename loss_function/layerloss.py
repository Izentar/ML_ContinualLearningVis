import torch
from  model.statistics.base import ModuleStatData, get_hash

class LayerBase():
    def __init__(self, device) -> None:
        self.device = device
        self.new_cl = False
        
    def set_current_class(self, cl):
        if(isinstance(cl, torch.TensorType)):
            self.current_cl = cl.item()
        else:
            self.current_cl = cl
        self.new_cl = True

class MeanNorm(LayerBase):
    def __init__(self, device, del_cov_after=False, scaling=0.01) -> None:
        super().__init__(device=device)

        self.loss_dict = {}
        self.scaling = scaling
        self.del_cov_after = del_cov_after
        self._name_gather_check = {}
        print(f'LAYERLOSS::MEAN_NORM: Scaling {self.scaling}')    
    
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
            loss_dict_key = (full_name, output.shape[1:])
            if(loss_dict_key in self.loss_dict):
                raise Exception(f'Loss for "{loss_dict_key}" was not processed. Tried to override loss.')
            self.loss_dict[loss_dict_key] = self.scaling * torch.sum(
                    torch.diag(torch.linalg.multi_dot((mean_diff, cov_inverse, mean_diff.T)))
                ).to(input[0].device)
        return module.register_forward_hook(inner)
        
    def gather_loss(self, loss) -> torch.Tensor:
        if(len(self.loss_dict) == 0):
            raise Exception("Loss dict is empty. Maybe tried to hook to the nonexistent layer?")
        sum_loss = torch.sum(torch.stack(list(self.loss_dict.values())))
        self.loss_dict = {}
        return loss + sum_loss

class DeepInversionFeatureLoss(LayerBase):
    '''
        Implementation of the forward hook to track feature statistics and compute a loss on them.
        Will compute mean and variance, and will use l2 as a loss
    '''
    def __init__(self, scale) -> None:
        self.r_feature = None
        self.scale = scale

    def hook_fun(self, module:torch.nn.Module, name:str, tree_name:str, new_tree_name:str):
        def inner(module:torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
            # hook co compute deepinversion's feature distribution regularization
            nch = input[0].shape[1]

            mean = input[0].mean([0, 2, 3])
            var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

            # forcing mean and variance to match between two distributions
            # other ways might work better, e.g. KL divergence
            self.r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(
                module.running_mean.data.type(var.type()) - mean, 2)
            # must have no output
        return module.register_forward_hook(inner)

    def gather_loss(self, loss):
        return loss + self.scale * self.r_feature 

class LayerGradPruning(LayerBase):
    def __init__(self, device, percent) -> None:
        super().__init__(device=device)
        
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

        return module.register_full_backward_hook(inner)

class LayerGradActivationPruning(LayerBase):
    def __init__(self, device, percent) -> None:
        super().__init__(device=device)
        
        self.percent = percent
        self.by_class  = dict()

    def hook_fun(self, module:torch.nn.Module, name:str, tree_name, new_tree_name):
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

        return module.register_full_backward_hook(inner)