import torch
from utils.data_manipulation import select_class_indices_tensor
from torch.utils.data import DataLoader
from config.default import tmp_stat_folder
from pathlib import Path
from config.default import model_to_save_file_type
from utils.utils import hook_model, get_model_hierarchy
import wandb
from collections.abc import Sequence
import pickle
import os
import time
from utils import pretty_print as pp
from torch import autocast

def get_hash(k, v:torch.Size):
    return (v, k)

def _get_shape_hash(k, v:torch.Tensor):
    return get_hash(k=k, v=v.shape)

def unhook(handles:dict|list):
    if(isinstance(handles, dict)):
        for h in handles.values():
            h.remove()
    elif(isinstance(handles, Sequence)):
        for h in handles:
            h.remove()
    else:
        raise Exception(f'Unsuported type: {type(handles)}')


class LazyDiskDict():
    # https://stackoverflow.com/questions/4014621/a-python-class-that-acts-like-dict
    def __init__(self, root_folder=tmp_stat_folder) -> None:
        super().__init__()
        self.root_folder = Path(root_folder)
        self.clear()

    def __setitem__(self, key, value) -> None:
        if(key in self._flushed_names):
            self._disk_remove_file(k=key)
        else:
            self._counter += 1
        self._in_memory_vals[key] = value
        self._all_keys[key] = self._counter

    def __getitem__(self, key):
        if(key not in self._all_keys): # check for possible key even if flushed
            raise KeyError((key, self._all_keys))
        try:
            return self._in_memory_vals[key] 
        except KeyError:
            self._disk_restore(k=key)
            return self._in_memory_vals[key]

    def __del__(self):
        for k in list(self._flushed_names.keys()):
            self._disk_remove_file(k=k)

    def __delitem__(self, key):
        self._disk_remove_file(k=key)
        del self._in_memory_vals[key]
        del self._all_keys[key]
        self._counter -= 1

    def __iter__(self):
        self._disk_restore_all()
        return iter(self._in_memory_vals)

    def __len__(self):
        return self._counter

    def __repr__(self):
        self._disk_restore_all()
        return repr(self._in_memory_vals)
    
    def __str__(self):
        return repr(self)

    def clear(self):
        self._in_memory_vals = {}
        self._all_keys = {} # to raise KeyError if not present
        self._flushed_types = {}
        self._flushed_names = {}
        self._counter = 0

    def has_key(self, key):
        return key in self._all_keys

    def keys(self):
        return self._all_keys.keys()

    def values(self):
        self._disk_restore_all()
        return self._in_memory_vals.values()

    def items(self):
        self._disk_restore_all()
        #exit()
        return self._in_memory_vals.items()

    def _disk_restore_all(self):
        for k in list(self._flushed_names.keys()):
            self._disk_restore(k=k)

    def _disk_flush(self, k):
        self._flushed_names[k] = f'{str(hash((self._in_memory_vals[k], time.time_ns, k)))}.lazyval'
        path = self.root_folder / self._flushed_names[k]
        with open(path, 'wb') as f:
            if(isinstance(self._in_memory_vals[k], torch.nn.Module) or isinstance(self._in_memory_vals[k], torch.Tensor)):
                torch.save(self._in_memory_vals[k], f=f)
                self._flushed_types[k] = 'pytorch'
            else:
                pickle.dump(self._in_memory_vals[k], file=f, protocol=pickle.HIGHEST_PROTOCOL)
                self._flushed_types[k] = 'pickle'
        del self._in_memory_vals[k]

    def _disk_restore(self, k) -> None:
        path = self.root_folder / self._flushed_names[k]
        with open(path, 'rb') as f:
            match(self._flushed_types[k]):
                case 'pytorch':
                    self._in_memory_vals[k] = torch.load(f=f)
                case 'pickle':
                    self._in_memory_vals[k] = pickle.load(file=f)
                case _:
                    raise Exception(f'Unknown type: {self._flushed_types[k]}')
        self._disk_remove_file(k=k)

    def _disk_remove_file(self, k):
        if(k in self._flushed_names.keys()):
            path = self.root_folder / self._flushed_names[k]
            if os.path.exists(path):
                os.remove(path=path)
            del self._flushed_names[k]
            del self._flushed_types[k]

    def flush(self, key=None):
        """
            Flush variable to disk
        """
        if(key is None):
            for k in list(self._in_memory_vals.keys()):
                self._disk_flush(k=k)
        else:
            self._disk_flush(k=key)

class ModuleStatData(torch.nn.Module):
    def __init__(self, full_name, lazydict=True) -> None:
        super().__init__()
        self._selected_dict_class = dict
        if(lazydict):
            self._selected_dict_class = LazyDiskDict
        self._cov:LazyDiskDict|dict|None = None
        self._cov_inverse:LazyDiskDict|dict|None = None
        self._M2n:LazyDiskDict|dict|None = None
        self._mean:LazyDiskDict|dict|None = None
        self._old_mean:LazyDiskDict|dict|None = None

        self._M2n_channel:LazyDiskDict|dict|None = None # for variance and std. Channel means classes
        self._mean_channel:LazyDiskDict|dict|None = None # # for variance and std. Channel means classes
        self._old_mean_channel:LazyDiskDict|dict|None = None

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
    def mean_channel(self):
        return self._mean_channel

    @mean_channel.setter
    def mean_channel(self, x):
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
    def var(self):
        if(self._M2n is None):
            return None
        ret = dict()
        for k, v in self._M2n.items():
            ret[k] = torch.div(v, self._counter_update - 1)
        return ret 
    
    @var.setter
    def var(self, x):
        raise Exception('No setter')    
    
    @property
    def std_channel(self):
        # unbiased version
        if(self._M2n_channel is None):
            return None
        ret = dict()
        for k, v in self._M2n_channel.items():
            ret[k] = torch.sqrt(torch.div(v, self._counter_update - 1))
        return ret
    
    @std_channel.setter
    def std_channel(self, x):
        raise Exception('No setter')
    
    @property
    def var_channel(self):
        if(self._M2n_channel is None):
            return None
        ret = dict()
        for k, v in self._M2n_channel.items():
            ret[k] = torch.div(v, self._counter_update - 1)
        return ret 
    
    @var_channel.setter
    def var_channel(self, x):
        raise Exception('No setter')    

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, x):
        raise Exception('No setter')

    def cov_inverse(self, del_cov_after=False, recalculate=False, failed_inverse_multiplication=8000, lazyload=True):
        if(self._cov_inverse is not None and not recalculate):
            return self._cov_inverse

        if(del_cov_after and recalculate):
            raise Exception("Bad flags. Cannot both be true: del_cov_after, recalculate")

        self._cov_inverse = self._selected_dict_class()
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
        if(lazyload and hasattr(self._cov_inverse, 'flush')):
            self._cov_inverse.flush()
        return self._cov_inverse

    def lazy_flush(self):
        self._cov.flush() if self._cov is not None and hasattr(self._cov, 'flush') else None
        self._cov_inverse.flush() if self._cov_inverse is not None and hasattr(self._cov_inverse, 'flush') else None
        self._M2n.flush() if self._M2n is not None and hasattr(self._M2n, 'flush') else None
        self._mean.flush() if self._mean is not None and hasattr(self._mean, 'flush') else None
        self._old_mean.flush() if self._old_mean is not None and hasattr(self._old_mean, 'flush') else None

class ModuleStat():
    def __init__(self, device:str, full_name:str, type:list[str]=None, shared_ref=None, flush_to_disk:str|bool=None) -> None:
        # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance
        self.shared_ref = dict() if shared_ref is None else shared_ref
        self.device = device
        self.data = ModuleStatData(full_name=full_name)

        # should be faster than any other solution
        self._update_call_mean = lambda x: None
        self._update_call_std = lambda x: None
        self._update_call_cov = lambda x: None
        if(type is None):
            type = ['mean', 'std']

        self.flush_to_disk_flag = bool((isinstance(flush_to_disk, bool) and flush_to_disk) or isinstance(flush_to_disk, str))

        self._flush_to_file_caller = self._flush_to_file if self.flush_to_disk_flag else lambda: None
        self._restore_from_file_caller = self._restore_from_file if self.flush_to_disk_flag else lambda: None

        self.tmp_full_path = tmp_stat_folder if flush_to_disk is None or (isinstance(flush_to_disk, bool) and flush_to_disk) else flush_to_disk
        if self.flush_to_disk_flag:
            Path(self.tmp_full_path).mkdir(parents=True, exist_ok=True)
            self.tmp_full_path = Path(self.tmp_full_path) / full_name

        self._update_call_mean = lambda x, y: None
        self._update_call_std = lambda x, y: None
        self._update_call_cov = lambda x, y: None
        self._update_call_mean_channel = lambda x, y: None
        self._update_call_std_channel = lambda x, y: None
        for u in type:
            match u:
                case 'mean':
                    self._update_call_mean = self._update_mean_first_call
                case 'std':
                    self._update_call_std = self._update_std_first_call
                    self._update_call_mean = self._update_mean_first_call
                case 'cov':
                    self._update_call_cov = self._update_cov_first_call
                    self._update_call_mean = self._update_mean_first_call
                case 'mean_channel':
                    self._update_call_mean_channel = self._update_mean_channel_first_call
                case 'std_channel':
                    self._update_call_std_channel = self._update_std_channel_first_call
                    self._update_call_mean_channel = self._update_mean_channel_first_call
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
        output = output.detach().to(self.device)
        values = self._split_by_target(output)
        for k, val in values.items():
            # iterate over classes
            for v in val:
                # iterate over values one by one
                self.data._counter_update += 1
                self._update_call_mean(k, v)
                self._update_call_std(k, v)
                self._update_call_mean_channel(k, v)
                self._update_call_std_channel(k, v)
                self._update_call_cov(k, v)

        self._flush_to_file_caller()

    def push(self, output):
        """
            Push data that will be used to gather stats.
        """
        self._update(output=output)
        #self.output_list.append(output)

    def register_batched_class_list(self, target_list:torch.Tensor):
        """
            Used to pass targets to this class.
        """
        self.shared_ref['target'] = target_list.to(self.device) # pass by reference

    def _split_by_target(self, torch_data) -> dict:
        """
            Split by target using shared_ref to 'indices_by_target' where dict[class idx, indices from target].
            Returns dict[class idx, selected data]
        """
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
    # calculates mean in the same size as input data
    def _update_mean_first_call(self, k:int, v:torch.Tensor):
        self.data._mean = self.data._selected_dict_class()
        self.data._old_mean = self.data._selected_dict_class()
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
    #####     MEAN CHANNEL     #####
    ################################
    # calculated mean by each channel. Output has size of channel dimension for given module
    def _update_mean_channel_first_call(self, k:int, v:torch.Tensor):
        self.data._mean_channel = self.data._selected_dict_class()
        self.data._old_mean_channel = self.data._selected_dict_class()
        self._update_mean_channel_first_occurence(k=k, v=v)
        
        self._update_call_mean_channel = self._update_mean_channel

    def _update_mean_channel_first_occurence(self, k:int, v:torch.Tensor):
        h = _get_shape_hash(k=k, v=v)
        # v of size (chn, x, y)
        v_chn = v.mean((1, 2))
        self.data._mean_channel[h] = v_chn.clone()
        self.data._old_mean_channel[h] = v_chn.clone()
       
    def _update_mean_channel_inner(self, k:int, v:torch.Tensor):
        h = _get_shape_hash(k=k, v=v)
        del self.data._old_mean_channel[h]
        v_chn = v.mean((1, 2))
        self.data._old_mean_channel[h] = self.data._mean_channel[h].clone()
        self.data._mean_channel[h] += (v_chn - self.data._mean_channel[h]) / self.data.counter_update

    def _update_mean_channel(self, k:int, v:torch.Tensor):
        try:
            self._update_mean_channel_inner(k=k, v=v)
        except KeyError:
            self._update_mean_channel_first_occurence(k=k, v=v)

    def calc_mean_channel(self) -> dict:
        return self.data.mean_channel

    ################################
    #####      STD / VAR       #####
    ################################
    # calculates std and var in the same size as input data
    def _update_std_first_call(self, k:int, v:torch.Tensor):
        self.data._M2n = self.data._selected_dict_class()
        self._update_std_first_occurence(k=k, v=v)

        self._update_call_std = self._update_std

    def _update_std_first_occurence(self, k:int, v:torch.Tensor):
        h = _get_shape_hash(k=k, v=v)
        self.data._M2n[h] = torch.ones(v.shape, device=v.device, requires_grad=False, dtype=v.dtype)

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
    
    def calc_var(self) -> dict:
        return self.data.var
    
    ################################
    #####   STD / VAR CHANNEL  #####
    ################################
    # calculated std and var by each channel. Output has size of channel dimension for given module
    def _update_std_channel_first_call(self, k:int, v:torch.Tensor):
        self.data._M2n_channel = self.data._selected_dict_class()
        self._update_std_channel_first_occurence(k=k, v=v)

        self._update_call_std_channel = self._update_std_channel

    def _update_std_channel_first_occurence(self, k:int, v:torch.Tensor):
        h = _get_shape_hash(k=k, v=v)
        # v of size (chn, x, y)
        v_chn = v.view((v.shape[0], -1)).var(1, correction=0)
        self.data._M2n_channel[h] = torch.ones(v_chn.shape, device=v_chn.device, requires_grad=False, dtype=v_chn.dtype)

    def _update_std_channel_inner(self, k:int, v:torch.Tensor):
        h = _get_shape_hash(k=k, v=v)
        v_chn = v.view((v.shape[0], -1)).var(1, correction=0)
        self.data._M2n_channel[h] += (v_chn - self.data._old_mean_channel[h]) * (v_chn - self.data._mean_channel[h])

    def _update_std_channel(self, k:int, v:torch.Tensor):
        try:
            self._update_std_channel_inner(k=k, v=v)
        except KeyError:
            self._update_std_channel_first_occurence(k=k, v=v)

    def calc_std_channel(self) -> dict:
        return self.data.std_channel
    
    def calc_var_channel(self) -> dict:
        return self.data.var_channel

    ################################
    #####         COV          #####
    ################################
    def _update_cov_first_call(self, k:int, v:torch.Tensor):
        self.data._cov = self.data._selected_dict_class()
        self.data._cov_diff = self.data._selected_dict_class()
        self.data._cov_diff_matrix = self.data._selected_dict_class()
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
        
    def get_const_data(self, lazyload=True) -> ModuleStatData:
        self._restore_from_file_caller()
        if(lazyload):
            self.data.lazy_flush()
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
    def __init__(self, model:torch.nn.Module, device, hook_verbose:bool=False, flush_to_disk=None, 
                 hook_to:list[str]=None, type:list[str]=None) -> None:
        super().__init__()
        self.deleted = False
        self.layers = dict()
        self.device = device
        self.shared_ref = dict()
        self.shared_ref['target'] = []
        self.shared_ref['val_by_target'] = dict()
        self.flush_to_disk = flush_to_disk
        self.type = type

        # hook to given layerss fun which creates ModuleStat and hook that pushes values
        # at some layers to this ModuleStat
        f = lambda layer, name, tree_name, full_name: layer.register_forward_hook(
            self._set_statistics_f(layer=layer, name=name, tree_name=tree_name, full_name=full_name)
        )
        self.handles = hook_model(model=model, fun=f, hook_to=hook_to, verbose=hook_verbose)

    def _set_statistics_f(self, layer, name, tree_name, full_name):
        full_name = f"{tree_name}.{name}" if len(tree_name) != 0 else name
        stat = ModuleStat(device=self.device, full_name=full_name, shared_ref=self.shared_ref, flush_to_disk=self.flush_to_disk,
            type=self.type)
        self.layers[full_name] = stat

        def inner(module, input, output):
            stat.push(output=output)
        return inner

    def get_cov(self, name:str):
        return self.layers[name].calc_cov()

    def get_mean(self, name:str):
        return self.layers[name].calc_mean()   

    def get_stats(self, names:list[str]=None, prepare=False) -> dict:
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
        """
            Used to pass targets to this class.
            It also creates dict[given target class, indices of given class in target_list]
        """
        self.shared_ref['target'] = target_list.to(self.device) # pass by reference
        self.shared_ref['indices_by_target'] = self._indices_by_target()

    def set_layer_stats_from(self, loaded:dict, strict=True):
        for k in self.layers.keys():  
            tmp = loaded.get(k)
            if(tmp is not None):
                self.layers[k].data = tmp
            elif(strict):
                raise Exception(f'Could not find key {k} in model {self.trainer.lightning_module.name}. Used "strict" flag.')

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
    stats_type:list[str]=None,
) -> dict|torch.Tensor:
    """
        Run a single evaluation loop with given single_dataloader. 
        Collect model stats and return tuple 
            - model layer stats - dictionary of all collected statistics (stats_type)
            - target tensor - all targets given by single_dataloader in order of appearing. 
        If there is little to no memory, try to push model into the device where the results are stored. For example if mean is stored on cpu,
        then model should be on cpu too to minimize memory footprint. 

        hook_to - list of all layers where it should collect statistics. The names should have a structure of a tree with '.' separator.
        
        stats_type:
            - if None - collect only 'mean' and 'std'.
            - 'mean', 'std' - collect mean and std for given layers.
            - 'cov' - collect covariance matrix.
            - 'mean_channel', 'std_channel' - collect mean and std for given layers with additional grouping by channel. 
                There will be additional indexing where you can choose channel.
    """
    model_layer_stats_obj = ModelLayerStatistics(model=model, device=device, hook_verbose=hook_verbose, flush_to_disk=flush_to_disk, 
                                                 hook_to=hook_to, type=stats_type)
    target_list = []

    model.eval()
    iterat = len(single_dataloader) if not fast_dev_run else fast_dev_run_max_batches
    progress_bar.setup_progress_bar(key='stats', text="[bright_red]Collect stats...", iterations=iterat)
    with torch.no_grad():
        for idx, (input, target) in enumerate(single_dataloader):
            if(fast_dev_run and idx == fast_dev_run_max_batches):
                break
            
            with autocast(device_type=device, dtype=torch.float16, enabled=False):
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
    try:
        out_by_layer_class = dict()
        for k_layer, v_layer in data.items():
            cov = v_layer.get_const_data().cov
            out_by_layer_class[k_layer] = dict()
            for k, v in cov.items():
                _, out, _ = torch.pca_lowrank(v, center=overestimated_rank)
                out_by_layer_class[k_layer][k] = out
        return out_by_layer_class
    except Exception as e:
        pp.sprint(f"{pp.COLOR.WARNING}WARNING: PCA not calculated. Exception:\n\t{e}")
        return None
