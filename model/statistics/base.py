import torch
from utils.data_manipulation import select_class_indices_tensor
from torch.utils.data import DataLoader
from config.default import tmp_stat_folder
from pathlib import Path
from config.default import model_to_save_file_type
from utils.utils import hook_model, get_model_hierarchy
import wandb
from collections.abc import Sequence

def _get_shape_hash(k, v:torch.Tensor):
    return (v.shape, k)

def _get_hash(k, v:torch.Size):
    return (v, k)

def unhook(handles:dict|list):
    if(isinstance(handles, dict)):
        for h in handles.values():
            h.remove()
    elif(isinstance(handles, Sequence)):
        for h in handles:
            h.remove()
    else:
        raise Exception(f'Unsuported type: {type(handles)}')

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

    def cov_inverse(self, del_cov_after=False, recalculate=False, failed_inverse_multiplication=8000):
        if(self._cov_inverse is not None and not recalculate):
            return self._cov_inverse

        if(del_cov_after and recalculate):
            raise Exception("Bad flags. Cannot both be true: del_cov_after, recalculate")

        self._cov_inverse = dict()
        for k, v in self.cov.items():
            try:
                self._cov_inverse[k] = torch.inverse(v)
            except RuntimeError as e:
                msg = f"ERROR: inverse for covariance matrix does not exist. Applying one times {failed_inverse_multiplication} as inverse covariance. Exception:\n\t{str(e)}\nMatrix:\n\t{v.shape}"
                print(msg)
                wandb.log({f'message/inverse_cov/key_{k}': msg})
                self._cov_inverse[k] = torch.ones_like(v) * failed_inverse_multiplication
            if(del_cov_after):
                del v
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
            Path(self.tmp_full_path).mkdir(parents=True, exist_ok=True)
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

    def _flush_to_file(self, handler=None):
        if(handler is None):
            handler = self.file_handler
        handler.seek(0, 0)
        torch.save(self.data, f=handler)
        del self.data

    def _restore_from_file(self, handler=None):
        if(handler is None):
            handler = self.file_handler
        handler.seek(0, 0)
        self.data = torch.load(f=handler)
    
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
        h = _get_shape_hash(k=k, v=v)
        self.data._mean[h] = v.clone()
        self.data._old_mean[h] = self.data._mean[h].clone()
       
    def _update_mean_inner(self, k:int, v:torch.Tensor):
        h = _get_shape_hash(k=k, v=v)
        del self.data._old_mean[h]
        self.data._old_mean[h] = self.data._mean[h].clone()
        self.data._mean[h] += (v - self.data._mean[h]) / self.data.counter_update

    def _update_mean(self, k:int, v:torch.Tensor):
        try:
            self._update_mean_inner(k=k, v=v)
        except KeyError:
            self._update_mean_first_occurence(k=k, v=v)

    def calc_mean(self) -> dict:
        return self.data.mean

    ################################
    #####         STD          #####
    ################################
    def _update_std_first_call(self, k:int, v:torch.Tensor):
        self.data._M2n = dict()
        self._update_std_first_occurence(k=k, v=v)

        self._update_call_std = self._update_std

    def _update_std_first_occurence(self, k:int, v:torch.Tensor):
        h = _get_shape_hash(k=k, v=v)
        self.data._M2n[h] = torch.zeros(v.shape[0], device=v.device, requires_grad=False, dtype=v.dtype)

    def _update_std_inner(self, k:int, v:torch.Tensor):
        h = _get_shape_hash(k=k, v=v)
        self.data._M2n[h] += (v - self.data._old_mean[h]) * (v - self.data._mean[h])

    def _update_std(self, k:int, v:torch.Tensor):
        try:
            self._update_std_inner(k=k, v=v)
        except KeyError:
            self._update_std_first_occurence(k=k, v=v)

    def calc_std(self) -> dict:
        return self.data.std

    ################################
    #####         COV          #####
    ################################
    def _update_cov_first_call(self, k:int, v:torch.Tensor):
        self.data._cov = dict()
        self.data._cov_diff = dict()
        self.data._cov_diff_matrix = dict()
        self._update_cov_first_occurence(k=k, v=v)

        self._update_call_cov = self._update_cov

    def _update_cov_first_occurence(self, k:int, v:torch.Tensor):
        h = _get_shape_hash(k=k, v=v)
        self.data._cov[h] = torch.zeros(v.shape[0], v.shape[0], device=self.device, dtype=v.dtype, requires_grad=False)
        self.data._cov_diff[h] = torch.zeros_like(self.data._old_mean[h], requires_grad=False).unsqueeze(dim=1)
        self.data._cov_diff_matrix[h] = torch.zeros(
            self.data._old_mean[h].shape[0], self.data._old_mean[h].shape[0], 
            device=self.device, 
            dtype=self.data._old_mean[h].dtype, 
            requires_grad=False
        )

    def _update_cov_inner(self, k:int, v:torch.Tensor):
        h = _get_shape_hash(k=k, v=v)
        torch.subtract(v.unsqueeze(dim=1), self.data._old_mean[h].unsqueeze(dim=1), out=self.data._cov_diff[h])
        torch.matmul(self.data._cov_diff[h], self.data._cov_diff[h].T, out=self.data._cov_diff_matrix[h])
        self.data._cov_diff_matrix[h].div_(self.data.counter_update)
        
        cov:torch.Tensor = self.data._cov[h]
        cov.mul_((self.data.counter_update - 2) / (self.data.counter_update - 1))
        cov.add_(self.data._cov_diff_matrix[h])

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
        
    def get_const_data(self) -> ModuleStatData:
        self._restore_from_file_caller()
        return self.data

    def save(self, path_filename:str|Path):
        if(not isinstance(path_filename, Path)):
            path_filename = Path(path_filename)
        path_filename.parent.mkdir(parents=True, exist_ok=True)

        self._flush_to_file(handler=path_filename)

    def restore(self, path_filename:str|Path) -> None:
        if(not isinstance(path_filename, Path)):
            path_filename = Path(path_filename)
        if(not path_filename.parent.exists()):
            raise Exception(f'Path {path_filename.parent} does not exist. Cannot load file {path_filename.name}')

        self._restore_from_file(handler=path_filename)

    def load(self, path_filename:str|Path) -> None:
        """
            The same as self.restore(path_filename) - alias.
        """
        self.restore(path_filename=path_filename)

class ModelLayerStatistics(torch.nn.Module):
    def __init__(self, model:torch.nn.Module, device, hook_verbose:bool=False, flush_to_disk=None, hook_to:list[str]=None) -> None:
        super().__init__()
        self.deleted = False
        self.layers = dict()
        self.device = device
        self.shared_ref = dict()
        self.shared_ref['target'] = []
        self.shared_ref['val_by_target'] = dict()
        self.flush_to_disk = flush_to_disk

        f = lambda layer, name, tree_name, new_tree_name: layer.register_forward_hook(
            self._set_statistics_f(layer=layer, name=name, tree_name=tree_name, new_tree_name=new_tree_name)
        )
        self.handles = hook_model(model=model, fun=f, hook_to=hook_to, verbose=hook_verbose)

    def _set_statistics_f(self, layer, name, tree_name, new_tree_name):
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
        #if(names is not None):
        #    for n in names:
        #        match n:
        #            case 'cov':
        #                for v in self.layers.values():
        #                    v.calc_cov()
        #            case 'mean':
        #                for v in self.layers.values():
        #                    v.calc_mean()
        #            case 'std':
        #                for v in self.layers.values():
        #                    v.calc_std()
        #            case _:
        #                raise Exception(f'Unknow name: {n}')
        #elif(prepare):
        #    for v in self.layers.values():
        #        v.calc_cov()
        #        v.calc_mean()
        #        v.calc_std()
        return self.layers

    def unhook(self):
        if(not self.deleted):
            for v in self.handles.values():
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

    def set_layer_stats_from(self, loaded:dict, strict=True):
        for k in self.layers.keys():  
            tmp = loaded.get(k)
            if(tmp is not None):
                self.layers[k].data = tmp
            elif(strict):
                raise Exception(f'Could not find key {k} in model {self.trainer.lightning_module.name()}. Used "strict" flag.')

    #def __setstate__(self, state):
    #    super().__setstate__(state)
    #    self.__dict__['layers'] = state['layers']
    #    self.__dict__['shared_ref'] = state['shared_ref']

def hook_model_stats(model:torch.nn.Module, stats: dict, fun, 
    hook_to:list[str]|list[torch.nn.Module|list]=None) -> list:
    """
        fun - has signature fun(module, full_name, layer_stat_data) that calls any module hook function and returns handle to it
    """
    if(stats is None):
        raise Exception('Constant statistics reference is "None".')
    handles = []
    already_hooked = []
    tree_name = ""
    ret = _hook_model_stats(
        model=model,
        stats=stats,
        fun=fun,
        tree_name=tree_name,
        handles=handles,
        hook_to=hook_to,
        already_hooked=already_hooked,
    )
    if(not isinstance(hook_to, bool) and set(already_hooked) != set(hook_to)):
        raise Exception(f'Not hooked to every provided layer. \n\tProvided: {hook_to}\n\tHooked to: {already_hooked}\n\tModel possible layers: {get_model_hierarchy(model=model)}')

    return ret

def _hook_model_stats(model:torch.nn.Module, stats: dict, fun, tree_name:str, handles, 
        hook_to:list[str]|list[torch.nn.Module|list], 
        already_hooked
    ) -> list:
    for name, module in model.named_children():
        new_tree_name = f"{tree_name}.{name}" if len(tree_name) != 0 else name
        
        if ((isinstance(hook_to, list) and 
                (new_tree_name in hook_to or module.__class__.__name__ in hook_to)) 
            or (isinstance(hook_to, bool) and hook_to)):
            layer_stat_data = stats[new_tree_name].get_const_data()
            handles.append(
                fun(module=module, full_name=new_tree_name, layer_stat_data=layer_stat_data)
            )
            to_append = new_tree_name
            if(isinstance(hook_to, list) and module.__class__.__name__ in hook_to):
                to_append = module.__class__.__name__
            already_hooked.append(to_append)
        
        _hook_model_stats(model=module, stats=stats, fun=fun, tree_name=new_tree_name, handles=handles, hook_to=hook_to, already_hooked=already_hooked)
    return handles

class LayerBase():
    def __init__(self, device) -> None:
        self.device = device
        self.current_batch_classes:torch.Tensor = None
        self.archived_batch_classes = None

    def set_current_batch_classes(self, classes):
        if(isinstance(classes, list)):
            classes = torch.tensor(classes)
            self.current_batch_classes = classes
        elif(isinstance(classes, torch.TensorType)):
            self.current_batch_classes = classes.clone()
        else:
            raise Exception(f'Unrecongized data type: "{type(classes)}"')
        self._set_archived(classes=classes)

    def _set_archived(self, classes:torch.Tensor):
        if(self.archived_batch_classes is None):
            self.archived_batch_classes = classes
        else:
            self.archived_batch_classes = torch.cat(self.archived_batch_classes, classes)
        
    def set_current_class(self, cl):
        if(isinstance(cl, torch.TensorType)):
            self.current_cl = cl.item()
        else:
            self.current_cl = cl

class LayerLoss(LayerBase):
    def __init__(self, device, del_cov_after=False, scaling=0.01) -> None:
        super().__init__(device=device)

        self.loss_list = []
        self.scaling = scaling
        self.del_cov_after = del_cov_after
        print(f'LAYER_LOSS: Scaling {self.scaling}')    
    
    def hook_fun(self, module:torch.nn.Module, full_name:str, layer_stat_data):
        def inner(module:torch.nn.Module, input:torch.Tensor, output:torch.Tensor):
            data:ModuleStatData = layer_stat_data
            h = _get_hash(k=self.current_cl, v=output.shape[1:])
            mean = data.mean[h].to(self.device)
            cov_inverse = data.cov_inverse(del_cov_after=self.del_cov_after)[h].to(self.device)
            output = output.view(output.shape[0], -1).to(self.device)

            mean_diff = output - mean
            self.loss_list.append(
                self.scaling * torch.sum(
                    torch.diag(torch.linalg.multi_dot((mean_diff, cov_inverse, mean_diff.T)))
                ).to(input[0].device)
            )
        return module.register_forward_hook(inner)
        
    def gather_loss(self, loss) -> torch.Tensor:
        if(len(self.loss_list) == 0):
            raise Exception("Loss list is empty. Maybe tried to hook to the nonexistent layer?")
        sum_loss = torch.sum(torch.stack(self.loss_list))
        self.loss_list = []
        return loss + sum_loss

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
            h = _get_hash(k=self.current_cl, v=grad_input[0].shape[1:])
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
    iterat = len(single_dataloader) if not fast_dev_run else fast_dev_run_max_batches
    progress_bar.setup_progress_bar(key='stats', text="[bright_red]Collect stats...", iterations=iterat)
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

def pca(data:dict[ModuleStatData], overestimated_rank=6):
    out_by_layer_class = dict()
    for k_layer, v_layer in data.items():
        cov = v_layer.get_const_data().cov
        out_by_layer_class[k_layer] = dict()
        for k, v in cov.items():
            _, out, _ = torch.pca_lowrank(v, center=overestimated_rank)
            out_by_layer_class[k_layer][k] = out
    return out_by_layer_class
