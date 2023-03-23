import torch
from utils.data_manipulation import select_class_indices_tensor
from torch.utils.data import DataLoader
from config.default import tmp_stat_folder
from pathlib import Path
from config.default import model_to_save_file_type



def hook_model(model:torch.nn.Module, fun, tree_name:str=None, handles=None, hook_to:list[str]|bool=False, verbose:bool=False) -> list:   
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
        new_tree_name = f"{tree_name}.{name}" if len(tree_name) != 0 else name
        if (isinstance(hook_to, list) and new_tree_name in hook_to) or (isinstance(hook_to, bool) and hook_to):
            handles.append(
                (
                    name,
                    tree_name,
                    module.register_forward_hook(fun(module, name=name, tree_name=tree_name))
                )
            )
        
        if(verbose):
            print(new_tree_name)
        hook_model(model=module, fun=fun, tree_name=new_tree_name, handles=handles, hook_to=hook_to, verbose=verbose)
    return handles

class ModuleStatData(torch.nn.Module):
    def __init__(self, full_name) -> None:
        super().__init__()
        self._cov:dict|None = None
        self._cov_inverse:dict|None = None
        self._M2n:dict|None = None
        self._mean:dict|None = None
        self._old_mean:dict|None = None

        self._counter_update = 0
        self._full_name = full_name

    @property
    def full_name(self):
        return self._full_name

    @full_name.setter
    def full_name(self, x):
        raise Exception('No setter')

    @property
    def counter_update(self):
        return self._counter_update

    @counter_update.setter
    def counter_update(self, x):
        raise Exception('No setter')

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, x):
        raise Exception('No setter')

    @property
    def std(self):
        # unbiased version
        if(self._M2n is None):
            return None
        ret = dict()
        for k, v in self._M2n.items():
            ret[k] = torch.sqrt(torch.div(v, self._counter_update - 1))
        return ret

    @std.setter
    def std(self, x):
        raise Exception('No setter')

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, x):
        raise Exception('No setter')

    def cov_inverse(self, del_cov_after=False, failed_inverse_multiplication=8000):
        if(self._cov_inverse is not None):
            return self._cov_inverse

        self._cov_inverse = dict()
        for k, v in self.cov.items():
            try:
                self._cov_inverse[k] = torch.inverse(v)
                if(del_cov_after):
                    del v
            except RuntimeError:
                print(f"ERROR: inverse for covariance matrix does not exist. Applying one times {failed_inverse_multiplication} as inverse covariance.")
                self._cov_inverse[k] = torch.ones_like(v) * failed_inverse_multiplication
        return self._cov_inverse

class ModuleStat():
    def __init__(self, device:str, full_name:str, to_update:list[str]=None, shared_ref=None, flush_to_disk:str|bool=None) -> None:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        self.shared_ref = dict() if shared_ref is None else shared_ref
        self.device = device
        self.data = ModuleStatData(full_name=full_name)
        #self.full_name = full_name

        #self._counter_update = 0
        #self._mean = None
        #self._M2n = None
        #self._cov = None

        # should be faster than any other solution
        self._update_call_mean = lambda x: None
        self._update_call_std = lambda x: None
        self._update_call_cov = lambda x: None
        if(to_update is None):
            to_update = ['mean', 'std', 'cov']

        self.flush_to_disk_flag = bool((isinstance(flush_to_disk, bool) and flush_to_disk) or isinstance(flush_to_disk, str))

        self._flush_to_file_caller = self._flush_to_file if self.flush_to_disk_flag else lambda: None
        self._restore_from_file_caller = self._restore_from_file if self.flush_to_disk_flag else lambda: None

        self.tmp_full_path = tmp_stat_folder if flush_to_disk is None or (isinstance(flush_to_disk, bool) and flush_to_disk) else flush_to_disk
        if self.flush_to_disk_flag:
            Path(self.tmp_full_path).mkdir(parents=True, exist_ok=True, mode=model_to_save_file_type)
            self.tmp_full_path = Path(self.tmp_full_path) / full_name

        for u in to_update:
            match u:
                case 'mean':
                    self._update_call_mean = self._update_mean_first_call
                case 'std':
                    self._update_call_std = self._update_std_first_call
                    self._update_call_mean = self._update_mean_first_call
                case 'cov':
                    self._update_call_cov = self._update_cov_first_call
                    self._update_call_mean = self._update_mean_first_call
                case _:
                    raise Exception(f'Unknown value: {u}')

        if(self.flush_to_disk_flag):
            self.file_handler = open(f'{self.tmp_full_path}.tmp.dat', 'w+b')
        self._flush_to_file_caller()

    def __del__(self):
        if(self.flush_to_disk_flag):
            self.file_handler.close()

    def _flush_to_file(self):
        self.file_handler.seek(0, 0)
        torch.save(self.data, f=self.file_handler)
        del self.data

    def _restore_from_file(self):
        self.file_handler.seek(0, 0)
        self.data = torch.load(f=self.file_handler)
    
    def _update(self, output:torch.Tensor):
        self._restore_from_file_caller()
        output = output.detach().view(output.shape[0], -1).to(self.device)
        values = self._split_by_target(output)
        for k, val in values.items():
            for v in val:
                self.data._counter_update += 1
                self._update_call_mean(k, v)
                self._update_call_std(k, v)
                self._update_call_cov(k, v)

        self._flush_to_file_caller()

    def push(self, output):
        self._update(output=output)
        #self.output_list.append(output)

    def register_batched_class_list(self, target_list:torch.Tensor):
        self.shared_ref['target'] = target_list.to(self.device) # pass by reference

    #def _set_new_output(self):
    #    if(self._new_output_list is None or self._new_output_list.shape[0] != len(self.output_list)):
    #        self._new_output_list = torch.cat(self.output_list)

    #def _should_recalc(self, to_check, index) -> bool:
    #    if(len(self.target_list) != len(self.output_list)):
    #        raise Exception(f"Class list size '{len(self.target_list)}' do not match output list size '{len(self.output_list)}'")
    #    if(to_check[0].get(index) is not None and to_check[0][index] != len(self.output_list)):
    #        to_check[0][index] = len(self.output_list)
    #        return True
    #    return False

    def _split_by_target(self, torch_data) -> dict:
        ret = dict()
        indices = self.shared_ref['indices_by_target']
        if(len(indices) == 0):
            raise Exception('Empty target list. Try to call "register_batched_class_list" before model.forward(input).')
        for cl, val in indices.items():
            ret[cl] = torch_data[val]
        return ret

    ################################
    #####         MEAN         #####
    ################################
    def _update_mean_first_call(self, k:int, v:torch.Tensor):
        self.data._mean = dict()
        self.data._old_mean = dict()
        self._update_mean_first_occurence(k=k, v=v)
        
        self._update_call_mean = self._update_mean

    def _update_mean_first_occurence(self, k:int, v:torch.Tensor):
        self.data._mean[k] = v.clone()
        self.data._old_mean[k] = self.data._mean[k].clone()

    def _update_mean_inner(self, k:int, v:torch.Tensor):
        del self.data._old_mean[k]
        self.data._old_mean[k] = self.data._mean[k].clone()
        self.data._mean[k] += (v - self.data._mean[k]) / self.data.counter_update

    def _update_mean(self, k:int, v:torch.Tensor):
        try:
            self._update_mean_inner(k=k, v=v)
        except KeyError:
            self._update_mean_first_occurence(k=k, v=v)

    def calc_mean(self) -> dict:
        return self.data.mean
        #self._set_new_output()
        #output_by_class = self._split_by_target(self._new_output_list)
        #ret = dict()
        #for k, v in output_by_class.items():
        #    if(self._should_recalc(self.mean, k)):
        #        ret[k())()] = torch.mean(v, dim=(0, ))
        #        self.mean[1][k] = ret.detach().cpu()
        #    else:
        #        ret[k] = self.mean[1][k].clone()
        #return ret

    ################################
    #####         STD          #####
    ################################
    def _update_std_first_call(self, k:int, v:torch.Tensor):
        self.data._M2n = dict()
        self._update_std_first_occurence(k=k, v=v)

        self._update_call_std = self._update_std

    def _update_std_first_occurence(self, k:int, v:torch.Tensor):
        self.data._M2n[k] = torch.zeros(v.shape[0], device=v.device, requires_grad=False, dtype=v.dtype)

    def _update_std_inner(self, k:int, v:torch.Tensor):
        self.data._M2n[k] += (v - self.data._old_mean[k]) * (v - self.data._mean[k])

    def _update_std(self, k:int, v:torch.Tensor):
        try:
            self._update_std_inner(k=k, v=v)
        except KeyError:
            self._update_std_first_occurence(k=k, v=v)

    def calc_std(self) -> dict:
        return self.data.std
        # unbiased version
        #if(self.data._M2n is None):
        #    return None
        #ret = dict()
        #for k, v in self.data._M2n.items():
        #    ret[k] = torch.sqrt(torch.div(v, self._counter_update - 1))
        #return ret
        #self._set_new_output()
        #output_by_class = self._split_by_target(self._new_output_list)
        #ret = dict()
        #for k, v in output_by_class.items():
        #    if(self._should_recalc(self.std, k)):
        #        ret[k] = torch.std(v, dim=(0, ))
        #        self.std[1][k] = ret.detach().cpu()
        #    else:
        #        ret[k] = self.std[1][k].clone()
        #return ret

    ################################
    #####         COV          #####
    ################################
    def _update_cov_first_call(self, k:int, v:torch.Tensor):
        self.data._cov = dict()
        self._update_cov_first_occurence(k=k, v=v)

        self._update_call_cov = self._update_cov

    def _update_cov_first_occurence(self, k:int, v:torch.Tensor):
        self.data._cov[k] = torch.zeros(v.shape[0], v.shape[0], device=self.device, dtype=v.dtype, requires_grad=False)

    def _update_cov_inner(self, k:int, v:torch.Tensor):
        diff:torch.Tensor = (v - self.data._old_mean[k]).unsqueeze(dim=0)
        weight = (self.data.counter_update - 2) / (self.data.counter_update - 1)
        cov:torch.Tensor = self.data._cov[k]
        cov.mul_(weight).add_(diff * diff.T * self.data.counter_update)
        del diff

    def _update_cov(self, k:int, v:torch.Tensor):
        # https://stats.stackexchange.com/questions/310680/sequential-recursive-online-calculation-of-sample-covariance-matrix
        try:
            self._update_cov_inner(k=k, v=v)
        except KeyError:
            self._update_cov_first_occurence(k=k, v=v)

    def calc_cov(self) -> dict:
        """
            Returns covariance matrix 2D, where diagonal contains the variance of each variable.
        """
        return self.data.cov
        #self._set_new_output()
        #output_by_class = self._split_by_target(self._new_output_list)
        #ret = dict()
        #for k, v in output_by_class.items():
        #    if(self._should_recalc(self.cov, k)):
        #        shape = v.shape
        #        x = v.view(shape[0], -1)
#
        #        # transpose needed
        #        # https://pytorch.org/docs/stable/generated/torch.cov.html
        #        ret[k] = torch.cov(x.T)
        #        self.cov[1][k] = ret.detach().cpu()
        #    else:
        #        ret[k] = self.cov[1][k].clone()
        #return ret
        
    def get_const_data(self) -> ModuleStatData:
        self._restore_from_file_caller()
        return self.data

class ModelLayerStatistics():
    def __init__(self, model:torch.nn.Module, device, hook_verbose:bool=False, flush_to_disk=None, hook_to:list[str]=None) -> None:
        self.deleted = False
        self.layers = dict()
        self.device = device
        self.shared_ref = dict()
        self.shared_ref['target'] = []
        self.shared_ref['val_by_target'] = dict()
        self.flush_to_disk = flush_to_disk

        f = lambda layer, name, tree_name: self._set_statistics_f(layer=layer, name=name, tree_name=tree_name)
        self.handles = hook_model(model=model, fun=f, hook_to=hook_to, verbose=hook_verbose)

    def _set_statistics_f(self, layer, name, tree_name):
        full_name = f"{tree_name}.{name}" if len(tree_name) != 0 else name
        stat = ModuleStat(device=self.device, full_name=full_name, shared_ref=self.shared_ref, flush_to_disk=self.flush_to_disk)
        self.layers[full_name] = stat

        def inner(module, input, output):
            stat.push(output=output)
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
                    case 'std':
                        for v in self.layers.values():
                            v.calc_std()
                    case _:
                        raise Exception(f'Unknow name: {n}')
        elif(prepare):
            for v in self.layers.values():
                v.calc_cov()
                v.calc_mean()
                v.calc_std()
        return self.layers

    def unhook(self):
        if(not self.deleted):
            for _, _, v in self.handles:
                v.remove()
            self.deleted = True

    def __del__(self):
        self.unhook()

    def _indices_by_target(self) -> dict:
        ret = dict()
        target_list = self.shared_ref['target']
        if(len(target_list) == 0):
            raise Exception('Empty target list. Try to call "register_batched_class_list" before model.forward(input).')
        unique_cl = torch.unique(target_list)
        for cl in unique_cl:
            ret[cl.item()] = select_class_indices_tensor(cl, target_list)
            #ret[cl] = torch_data[indices]
        return ret

    def register_batched_class_list(self, target_list):
        self.shared_ref['target'] = target_list.to(self.device) # pass by reference
        self.shared_ref['indices_by_target'] = self._indices_by_target()


def hook_model_stats(model:torch.nn.Module, stats: dict, fun, tree_name:str=None, handles=None, hook_to:list[str]=None):
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
        new_tree_name = f"{tree_name}.{name}" if len(tree_name) != 0 else name
        
        if (isinstance(hook_to, list) and new_tree_name in hook_to) or (isinstance(hook_to, bool) and hook_to):
            layer_stat_data = stats[new_tree_name].get_const_data()
            handles.append(module.register_forward_hook(
                fun(full_name=new_tree_name, layer_stat_data=layer_stat_data)
            ))
        
        hook_model_stats(model=module, stats=stats, fun=fun, tree_name=new_tree_name, handles=handles, hook_to=hook_to)
    return handles

class LayerLossData():
    def __init__(self, full_name, scaling:torch.Tensor) -> None:
        self.full_name = full_name
        self.layer_stat_data = None
        self.scaling = scaling

    def hook_fun(self, module:torch.nn.Module, input, output:torch.Tensor, layer_stat_data):
        data:ModuleStatData = layer_stat_data
        mean = data.mean[self.current_cl]
        output = output.reshape(output.shape[0], -1).to(mean.device)

        mean_diff = output - mean
        try:
            return self.scaling * torch.sum(mean_diff * torch.inverse(data.cov) * mean_diff.T)
        except RuntimeError:
            print(f"ERROR: inverse for covariance matrix does not exist. Applying zero as loss for module: {module._get_name()}")
            return torch.zeros(1, dtype=output.dtype, requires_grad=output.requires_grad, device=output.device)

class LayerLoss():
    def __init__(self, device, del_cov_after=False) -> None:
        self.loss_list = []
        self.current_batch_classes:torch.Tensor = None
        self.archived_batch_classes = None
        self.scaling = torch.tensor(2, dtype=torch.float32)
        self.losses_data = dict()
        self.device = device
        self.del_cov_after = del_cov_after

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

    def set_current_class(self, cl):
        if(isinstance(cl, torch.TensorType)):
            self.current_cl = cl.item()
        else:
            self.current_cl = cl
            
    def hook_fun(self, full_name:str, layer_stat_data):
        #data = LayerLossData(full_name, self.scaling)
        #self.losses_data[full_name] = data

        def inner(module:torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
            data:ModuleStatData = layer_stat_data
            mean = data.mean[self.current_cl].to(self.device)
            cov_inverse = data.cov_inverse(del_cov_after=self.del_cov_after)[self.current_cl].to(self.device)
            output = output.view(output.shape[0], -1).to(self.device)

            mean_diff = output - mean
            self.loss_list.append(
                torch.sum(
                    self.scaling * torch.linalg.multi_dot((mean_diff, cov_inverse, mean_diff.T))
                ).to(input[0].device)
            )
        return inner
        
    def gather_loss(self, loss) -> torch.Tensor:
        if(len(self.loss_list) == 0):
            raise Exception("Loss list is empty")
        sum_loss = torch.sum(torch.stack(self.loss_list))
        self.loss_list = []
        return loss + sum_loss

def collect_model_layer_stats(
    model:torch.nn.Module, 
    single_dataloader, 
    device, 
    hook_verbose:bool=False, 
    progress_bar=None, 
    flush_to_disk=None, 
    hook_to:list[str]=None,
    fast_dev_run:bool=False,
    fast_dev_run_max_batches:int=30,
) -> dict|torch.Tensor:
    """
        Collect model stats and return tuple 
            - model layer stats
            - target tensor
        If there is little to no memory, try to push model into the device where the results are storen. For example if mean is stored on cpu,
        then model should be on cpu too to minimize memory footprint. 
    """
    model_layer_stats_obj = ModelLayerStatistics(model=model, device=device, hook_verbose=hook_verbose, flush_to_disk=flush_to_disk, hook_to=hook_to)
    target_list = []

    model.eval()
    #model = model.to('cpu')

    progress_bar.setup_progress_bar(key='stats', text="[bright_red]Collect stats...", iterations=len(single_dataloader) if fast_dev_run else fast_dev_run_max_batches)
    with torch.no_grad():
        for idx, (input, target) in enumerate(single_dataloader):
            if(fast_dev_run and idx == fast_dev_run_max_batches):
                break
            input = input.to(model.device)
            model_layer_stats_obj.register_batched_class_list(target)
            target = target.to(model.device)
            model(input)
            target_list.append(target)
            progress_bar.update(key='stats')
            progress_bar.refresh()
    
    progress_bar.clear(key='stats')
    target_list = torch.cat(target_list)
    model_stats = model_layer_stats_obj.get_stats(prepare=True)
    model_layer_stats_obj.unhook()

    return model_stats, target_list


def pca(data:dict[ModuleStatData]):
    out_by_layer_class = dict()
    for k_layer, v_layer in data.items():
        cov = v_layer.get_const_data().cov
        out_by_layer_class[k_layer] = dict()
        for k, v in cov.items():
            _, out, _ = torch.pca_lowrank(v)
            out_by_layer_class[k_layer][k] = out
    return out_by_layer_class
