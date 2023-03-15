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