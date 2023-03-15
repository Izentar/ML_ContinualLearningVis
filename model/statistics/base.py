import torch
from utils.data_manipulation import select_class_indices_tensor
from torch.utils.data import DataLoader

def hook_model(model:torch.nn.Module, fun, tree_name:str=None, handles=None) -> list:   
    """
        fun - fun with signature fun(layer, name, tree_name) that returns
            new fun with signature fun2(module, input, output)
        return tuple(name, tree_name, handle_to_layer)
    """
    if(handles is None):
        handles = []
    if(tree_name is None):
        tree_name = ""
    for name, module in model.named_children():
        handles.append(
            (
                name,
                tree_name,
                module.register_forward_hook(fun(module, name, tree_name))
            )
        )
        tree_name = f"{tree_name}.{name}"
        hook_model(model=module, fun=fun, tree_name=tree_name, handles=handles)
    return handles

class ModuleStatConstData():
    def __init__(self, module_stat) -> None:
        self.cov:dict = module_stat.calc_cov()
        self.std_mean:dict = module_stat.calc_std_mean()
        self.mean:dict = module_stat.calc_mean()

class ModuleStat():
    def __init__(self) -> None:
        self.output_list = []
        self.target_list = []
        self._new_output_list:list = None
        
        self.cov = [dict(), dict()]
        self.std_mean = [dict(), dict()]
        self.mean = [dict(), dict()]

    def push(self, output):
        self.output_list.append(output)

    def register_batched_class_list(self, class_list):
        self.target_list.extend(class_list)

    def register_full_class_list(self, target_list_full):
        self.target_list = target_list_full

    def _set_new_output(self):
        if(self._new_output_list is None or self._new_output_list.shape[0] != len(self.output_list)):
            self._new_output_list = torch.cat(self.output_list)

    def _should_recalc(self, to_check, index) -> bool:
        if(len(self.target_list) != len(self.output_list)):
            raise Exception(f"Class list size '{len(self.target_list)}' do not match output list size '{len(self.output_list)}'")
        if(to_check[0].get(index) is not None and to_check[0][index] != len(self.output_list)):
            to_check[0][index] = len(self.output_list)
            return True
        return False

    def _divide_to_classes(self, torch_data) -> dict:
        ret = dict()
        unique_cl = torch.unique(torch.tensor(self.target_list))
        for cl in unique_cl:
            indices = select_class_indices_tensor(torch.tensor(self.target_list), torch_data)
            ret[cl] = self._new_output_list[indices]
        return ret

    def calc_std_mean(self) -> dict:
        self._set_new_output()
        output_by_class = self._divide_to_classes(self._new_output_list)
        ret = dict()
        for k, v in output_by_class.items():
            if(self._should_recalc(self.std_mean, k)):
                ret[k] = torch.std_mean(v, dim=(0, ))
                self.std_mean[1][k] = ret.detach().cpu()
            else:
                ret[k] = self.std_mean[1][k].clone()
        return ret

    def calc_mean(self) -> dict:
        self._set_new_output()
        output_by_class = self._divide_to_classes(self._new_output_list)
        ret = dict()
        for k, v in output_by_class.items():
            if(self._should_recalc(self.mean, k)):
                ret[k] = torch.mean(v, dim=(0, ))
                self.mean[1][k] = ret.detach().cpu()
            else:
                ret[k] = self.mean[1][k].clone()
        return ret

    def calc_cov(self):
        """
            Returns covariance matrix 2D, where diagonal contains the variance of each variable.
        """
        self._set_new_output()
        output_by_class = self._divide_to_classes(self._new_output_list)
        ret = dict()
        for k, v in output_by_class.items():
            if(self._should_recalc(self.cov, k)):
                shape = v.shape
                x = v.view(shape[0], -1)

                # transpose needed
                # https://pytorch.org/docs/stable/generated/torch.cov.html
                ret[k] = torch.cov(x.T)
                self.cov[1][k] = ret.detach().cpu()
            else:
                ret[k] = self.cov[1][k].clone()
        return ret
        
    def get_const_data(self) -> ModuleStatConstData:
        return ModuleStatConstData(self)

class ModelLayerStatistics():
    def __init__(self, model:torch.nn.Module) -> None:
        self.deleted = False
        self.layers = dict()

        f = lambda layer, name, tree_name: self._set_statistics_f(layer=layer, name=name, tree_name=tree_name)
        self.handles = hook_model(model=model, fun=f)

    def _set_statistics_f(self, layer, name, tree_name):
        stat = ModuleStat()
        self.layers[f"{tree_name}.{name}"] = stat

        def inner(module, input, output):
            stat.push(output=output.detach().cpu())
        return inner

    def get_cov(self, name:str):
        return self.layers[name].calc_cov()

    def get_mean(self, name:str):
        return self.layers[name].calc_mean()   

    def get_stats(self, names:list[str]=None, prepare=False) -> dict:
        """
            Calculate and saves internally the results.
            After calling again, returns value faster than the first time.

            Returns ModuleStat reference objects in dictionary {tree.name: obj}
        """
        if(names is not None):
            for n in names:
                match n:
                    case 'cov':
                        for v in self.layers.values():
                            v.calc_cov()
                    case 'mean':
                        for v in self.layers.values():
                            v.calc_mean()
                    case 'std_mean':
                        for v in self.layers.values():
                            v.calc_std_mean()
                    case _:
                        raise Exception(f'Unknow name: {n}')
        elif(prepare):
            for v in self.layers.values():
                v.calc_cov()
                v.calc_mean()
                v.calc_std_mean()
        return self.layers

    def unhook(self):
        if(not self.deleted):
            for _, _, v in self.handles:
                v.remove()
            self.deleted = True

    def __del__(self):
        self.unhook()

    def register_full_class_list(self, target_list_full):
        for v in self.layers.values():
            v.register_full_class_list(target_list_full)

def hook_model_stats(model:torch.nn.Module, stats: dict, fun, tree_name:str=None, handles=None):
    """
        fun - has signature fun(module, input, output, layer_stat_data)
    """
    if(stats is None):
        raise Exception('Constant statistics reference is "None".')
    if(handles is None):
        handles = []
    if(tree_name is None):
        tree_name = ""
    for name, module in model.named_children():
        tree_name = f"{tree_name}.{name}"
        layer_stat_data = stats[tree_name].get_const_data()
        handles.append(module.register_forward_hook(
            lambda module, input, output: fun(module, input, output, layer_stat_data)
        ))
        
        hook_model_stats(model=module, stats=stats, fun=fun, tree_name=tree_name, handles=handles)
    return handles

class LayerLoss():
    def __init__(self) -> None:
        self.loss_list = []
        self.current_batch_classes:torch.Tensor = None
        self.archived_batch_classes = None
        self.loss_weight = torch.tensor(2, dtype=torch.float32)

    def _set_archived(self, classes:torch.Tensor):
        if(self.archived_batch_classes is None):
            self.archived_batch_classes = classes
        else:
            self.archived_batch_classes = torch.cat(self.archived_batch_classes, classes)

    def set_current_batch_classes(self, classes):
        if(isinstance(classes, list)):
            classes = torch.tensor(classes)
            self.current_batch_classes = classes
        elif(isinstance(classes, torch.TensorType)):
            self.current_batch_classes = classes.clone()
        else:
            raise Exception(f'Unrecongized data type: "{type(classes)}"')
        self._set_archived(classes=classes)

    def hook_fun(self, module:torch.nn.Module, input, output:torch.Tensor, layer_stat_data):
        data:ModuleStatConstData = layer_stat_data
        output = output.view(output.shape[0], -1)
        mean_diff = output - data.mean,
        try:
            return self.loss_weight * torch.sum(mean_diff * torch.inverse(data.cov) * mean_diff.T)
        except RuntimeError:
            print(f"ERROR: inverse for covariance matrix does not exist. Applying zero as loss for module: {module._get_name()}")
            return torch.zeros(1, dtype=output.dtype, requires_grad=output.requires_grad, device=output.device)
        
    def gather_loss(self) -> torch.Tensor:
        if(len(self.loss_list) == 0):
            raise Exception("Loss list is empty")
        return torch.sum(self.loss_list)

def collect_model_layer_stats(model:torch.nn.Module, single_dataloader) -> dict|torch.Tensor:
    """
        Collect model stats and return tuple 
            - model layer stats
            - target tensor
    """
    model_layer_stats = ModelLayerStatistics(model=model)
    target_list = []

    model.eval()

    with torch.no_grad():
        for idx, (input, target) in enumerate(single_dataloader):
            input = input.to(model.device)
            target = target.to(model.device)
            model.forward(input)
            target_list.append(target)
    
    target_list = torch.cat(target_list)
    model_layer_stats.register_full_class_list(target_list_full=target_list)
    model_stats = model_layer_stats.get_stats(prepare=True)
    model_layer_stats.unhook()

    return model_layer_stats, target_list
