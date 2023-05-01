from copy import deepcopy
from argparse import Namespace

def check(self_obj, args, cfg_map):
    if(args is None and cfg_map is None):
        raise Exception(f"No config provided.")
    if(cfg_map is not None):
        for k in cfg_map.keys():
            if(k not in self_obj.CONFIG_MAP):
                raise Exception(f"Unknown config map key: {k} from {self_obj.CONFIG_MAP.keys()}")

def setup_args(args_map, args, root:str):
    args = deepcopy(args)
    if(args_map is not None):
        for k, v in args_map.items():
            name = k.split('.')
            if(len(name) == 1):
                if(name[0] == ''):
                    setattr(args, root, v)
                else:
                    if(hasattr(args, root)):
                        obj = getattr(args, root)
                    else:
                        setattr(args, root, Namespace)
                        obj = getattr(args, root)
                    setattr(obj, name[0], v)
            else:
                obj = getattr(args, root)
                for n in name[: -1]:
                    if(hasattr(obj, n)):
                        obj = getattr(obj, n)
                    else:
                        setattr(obj, n, Namespace)
                        obj = getattr(obj, n)
                setattr(args, name[-1], v)

    #print(args)
            #if(len(name) == 1):
            #    map_to = self_obj.VAR_MAP.get('')
            #else:
            #    parent = '.'.join(name[: -1]) 
            #    map_to = self_obj.VAR_MAP.get(parent)
            #    if(map_to is not None):
            #        obj = getattr(self_obj, map_to)
            #        setattr(obj, name[-1], v)
            #else:
                # ignore for multiple inheritance
                #raise Exception(f"Bad value: {parent}. Only possible: {self_obj.VAR_MAP.keys()}")
    return args

def setup_cfg_map(self_obj, args, cfg_map):
    if(cfg_map is not None):
        for k, cfg_class in self_obj.CONFIG_MAP.items():
            name = k
            if(k not in cfg_map):
                if(args is None): # setup if not set
                    setattr(self_obj, name, cfg_class())
            else: # override
                setattr(self_obj, name, cfg_map[k])