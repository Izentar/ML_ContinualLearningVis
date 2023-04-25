import torch
from collections.abc import Sequence

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
        new_tree_name = f"{tree_name}.{name}" if len(tree_name) != 0 else name
        if ((isinstance(hook_to, list) and 
                (new_tree_name in hook_to or module.__class__.__name__ in hook_to)) 
            or (isinstance(hook_to, bool) and hook_to)):
            handles[new_tree_name] = fun(module, name=name, tree_name=tree_name, new_tree_name=new_tree_name)
            to_append = new_tree_name
            if(isinstance(hook_to, list) and module.__class__.__name__ in hook_to):
                to_append = module.__class__.__name__
            already_hooked.append(to_append)
        
        if(verbose):
            print(new_tree_name)
        _hook_model(model=module, fun=fun, tree_name=new_tree_name, handles=handles, hook_to=hook_to, verbose=verbose, already_hooked=already_hooked)
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

def get_obj_dict(obj, reject_from: object):
    return {k: v for k, v in vars(obj).items() if not isinstance(v, reject_from)}