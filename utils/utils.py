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
        replace_layer(childModule, f"{name}.{childName}", classToReplace, replaceWithClass_f) 

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