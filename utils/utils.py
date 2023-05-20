import torch
from collections.abc import Sequence

DEBUG = False

def replace_layer(module:torch.nn.Module, name:str, classToReplace, replaceWithClass_f):
    """
        module - torch Module
        name - source name, needed for recursion. Any string is valid.
        classToReplace - bare class like torch.nn.ReLU
        replaceWithClass - lambda expression or function with signature fun(parent_name, attr_name, source_attribute), where 
            * parent_name - names in the path to this attribure
            * name - name of the attribure
            * source_attribute is object you want to replace.
    """
    for attr_str in dir(module):
        target_attr = getattr(module, attr_str)
        if type(target_attr) == classToReplace:
            print(f'replaced: {name}.{attr_str}')
            setattr(module, attr_str, replaceWithClass_f(name, attr_str, target_attr))
    for childName, childModule in module.named_children():
        replace_layer(childModule, name=f"{name}.{childName}", classToReplace=classToReplace, replaceWithClass_f=replaceWithClass_f) 

def check_python_index(index:list[int]|int|None|bool, size:int, current_val:int, is_none_good=False) -> bool:
    def _check(index:str|bool|int, size, current_val):
        if(isinstance(index, str)):
            index = str_to_bool_int(index)
        if(isinstance(index, bool)):
            return index
        return bool(
            index == current_val or (index < 0 and size + index == current_val)
        )
            
    if(index is None):
        return is_none_good
    if(isinstance(index, bool)):
        return index
    if(isinstance(index, Sequence)):
        is_good = False
        if(len(index) == 0):
            return False
        for v in index:
            is_good = is_good or _check(index=v, size=size, current_val=current_val)
        return is_good
    return _check(index=index, size=size, current_val=current_val)

def check_python_enabled(index:list[int]|int|None|bool, is_none_good=False) -> bool:         
    if(index is None):
        return is_none_good
    if(isinstance(index, bool)):
        return index
    if(isinstance(index, list)):
        if(len(index) == 0):
            return False
        else:
            for k in index:
                if(check_python_enabled(k, is_none_good=is_none_good)):
                    return True
                return False
        return True
    return True

def str_to_bool_int(val:list[str]|str):
    def inner(string:str):
        if(isinstance(string, str)):
            b = str_is_true(string)
            if(b is not None):
                return b
            return int(string)
    
    if(isinstance(val, str)):
        return inner(val)
    elif(isinstance(val, Sequence)):
        ret = []
        for v in val:
            ret.append(inner(v))
        return ret
    elif(isinstance(val, bool)):
        return val
    else:
        raise Exception(f'Not int or bool: {val}')

def str_is_true(val:str) -> bool|None:
    if(val == 'true' or val == 't' or val == 'True' or val == 'y' or val == 'Y'):
        return True
    if(val == 'false' or val == 'f' or val == 'False' or val == 'n' or val == 'N'):
        return False
    return None

def parse_image_size(image_size):
    if(isinstance(image_size, Sequence)):
        if(len(image_size) == 3):
            channels = image_size[0]
            w = image_size[1]
            h = image_size[2]
        elif(len(image_size) == 2):
            channels = None
            w = image_size[0]
            h = image_size[1]
        elif(len(image_size) == 1):
            channels = None
            w = image_size[0]
            h = image_size[0]
        else:
            raise Exception(f'Wrong image size {image_size}')
    else:
        channels = None
        w = image_size
        h = image_size
    return channels, w, h

def hook_model(model:torch.nn.Module, 
        fun, 
        hook_to:list[str]|bool|torch.nn.Module|list[torch.nn.Module|list]=False, 
        verbose:bool=False
    ) -> list:  
    handles = dict()
    tree_name = ""
    already_hooked = []
    ret = _hook_model(
        model=model,
        fun=fun,
        hook_to=hook_to,
        verbose=verbose,
        handles=handles,
        tree_name=tree_name,
        already_hooked=already_hooked,
    )
    if(not isinstance(hook_to, bool) and set(already_hooked) != set(hook_to)):
        raise Exception(f'Not hooked to every provided layer. \n\tProvided: {hook_to}\n\tHooked to: {already_hooked}\n\tModel possible layers: {get_model_hierarchy(model=model)}')

    return ret

def _hook_model(model:torch.nn.Module, fun, tree_name:str, handles, already_hooked:list[str], 
    hook_to:list[str]|bool|torch.nn.Module|list[torch.nn.Module|list]=False, 
    verbose:bool=False) -> list:   
    """
        fun - fun with signature fun(layer, name, tree_name) that returns
            new fun with signature fun2(module, input, output)
        return tuple(name, tree_name, handle_to_layer)
    """
    for name, module in model.named_children():
        full_name = f"{tree_name}.{name}" if len(tree_name) != 0 else name
        if ((isinstance(hook_to, list) and 
                (full_name in hook_to or module.__class__.__name__ in hook_to)) 
            or (isinstance(hook_to, bool) and hook_to)):
            handles[full_name] = fun(module, name=name, tree_name=tree_name, full_name=full_name)
            to_append = full_name
            if(isinstance(hook_to, list) and module.__class__.__name__ in hook_to):
                to_append = module.__class__.__name__
            already_hooked.append(to_append)
        
        if(verbose):
            print(full_name)
        _hook_model(model=module, fun=fun, tree_name=full_name, handles=handles, hook_to=hook_to, verbose=verbose, already_hooked=already_hooked)
    return handles


def log_wandb_tensor_decorator(func, name:list[str], logger):
    """
        Log value. The name is passed in list of length 1 to keep the string reference in case of modifying it.
    """
    import wandb
    def inner(val:torch.Tensor):
        wandb.log({name[0]: val.item()})
        return func(val)
    return inner

def get_model_hierarchy(model, separator='.') -> list[str]:
    model_names = []
    tree_name = ""
    return _get_model_hierarchy(
        model=model,
        separator=separator,
        model_names=model_names,
        tree_name=tree_name,
    )

def _get_model_hierarchy(model, tree_name:str, model_names:list[str], separator='.') -> list[str]:
    for name, module in model.named_children():
        new_tree_name = f"{tree_name}{separator}{name}" if len(tree_name) != 0 else name
        model_names.append(new_tree_name)
        _get_model_hierarchy(model=module, separator=separator, tree_name=new_tree_name, model_names=model_names)
    return model_names

def get_obj_dict(args, reject_from:list=None, accept_only:list=None, recursive=False, recursive_types: list[object]=None):
    if(recursive_types is None and not recursive):
            raise Exception('Bad arguments. Cannot be "recursive" without "recursive_types"')
    ret = vars(args).copy()
    if(DEBUG):
        print('keys:', list(ret.keys()))
        print(accept_only)
    #if(reject_from is not None):
    #    ret = {k: v for k, v in ret.items() if k not in reject_from}
    if(accept_only is not None):
        for k in list(ret.keys()):
            if(k not in accept_only):
                ret.pop(k, None)

    if(recursive):
        if(recursive_types is None and not recursive):
            raise Exception('Bad arguments. Cannot be "recursive" without "recursive_types"')
        for k in list(ret.keys()):
            for r in recursive_types:
                if(isinstance(ret[k], r)):
                    if(DEBUG):
                        print('recursion', k, ret[k])
                    # should accept_only in first step of recursion.
                    ret[k] = get_obj_dict(ret[k], reject_from=None, accept_only=None, recursive=recursive, recursive_types=recursive_types)
    if(DEBUG):
        print('return', ret)
    return ret

def search_kwargs(kwargs: dict, vals:list[str]):
    new_kwargs = {}
    for k, v in kwargs.items():
        if(k in vals):
            new_kwargs[k] = v
    return new_kwargs

def rgetattr(obj, name:str, separator='.', nofound_is_ok=False) -> object|None:
    var_names = name.split(sep=separator)
    current = obj
    for v in var_names:
        try:
            current = getattr(current, v)
        except AttributeError:
            if(nofound_is_ok):
                return None
            raise Exception(f"Could not find '{v}', name '{name}' \nfor object: {vars(current)}\n\nroot obj '{obj}'")
    return current

def setup_obj_dataclass_args(src_obj, args, root_name, sep='.', recursive=False, recursive_types=None):
    tree = {}
    for k in src_obj.VAR_MAP.keys():
        k: str = k
        split = k.split('.')
        current = tree
        for s in split:
            current[s] = {}
            current = current[s]

    for k, v in src_obj.VAR_MAP.items():
        fn = src_obj.CONFIG_MAP[v].__init__
        if(DEBUG):
            print()
            print(src_obj.CONFIG_MAP[v], v)
        if(k == ''):
            setattr(src_obj, v, src_obj.CONFIG_MAP[v](
                **get_obj_dict(
                    rgetattr(args, f"{root_name}"), #reject_from=list(tree.keys()), 
                    accept_only=fn.__code__.co_varnames[:fn.__code__.co_argcount], 
                    recursive=recursive, recursive_types=recursive_types)
            ))
        else:
            #split = k.split('.')
            #current_tree = tree
            #for s in split:
            #    current_tree = current_tree[s]
            #current_tree = list(current_tree.keys())
#
            setattr(src_obj, v, src_obj.CONFIG_MAP[v](
                **get_obj_dict(
                    rgetattr(args, f"{root_name}{sep}{k}"), #reject_from=current_tree, 
                    accept_only=fn.__code__.co_varnames[:fn.__code__.co_argcount], 
                    recursive=recursive, recursive_types=recursive_types)
            ))

def check_cfg_var_maps(src_obj):
    for k, v in src_obj.VAR_MAP.items():
        try:
            v2 = src_obj.CONFIG_MAP[v]
        except KeyError:
            raise Exception(f"Bad maps. Could not find if CONFIG_MAP key {v} for VAR_MAP key {k}")
        
    vals = list(src_obj.VAR_MAP.values())
    for k in src_obj.CONFIG_MAP.keys():
        if k not in vals:
            raise Exception(f'Could not find in VAR_MAP value "{k}".')
        
def time_in_sec_format_to_hourly(seconds):
    sec = seconds % (24 * 3600)
    hour = sec // 3600
    sec %= 3600
    min = sec // 60
    sec %= 60
    return int(hour), int(min), int(sec)