import torch

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
    def _check(index:str|bool, size, current_val):
        index = int(index)
        return bool(
            index == current_val or (index < 0 and size + index == current_val)
        )
            
    if(index is None):
        return is_none_good
    if(isinstance(index, bool)):
        return index
    if(isinstance(index, list)):
        is_good = False
        if(len(index) == 0):
            return False
        for v in index:
            is_good = is_good or _check(index=v, size=size, current_val=current_val)
        return is_good
    return _check(index=index, size=size, current_val=current_val)

def str_is_true(val:str) -> bool|None:
    if(val == 'true' or val == 't' or val == 'True' or val == 'y' or val == 'Y'):
        return True
    if(val == 'false' or val == 'f' or val == 'False' or val == 'n' or val == 'N'):
        return False
    return None