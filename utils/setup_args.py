

def check(self_obj, args, cfg_map):
    if(args is None and cfg_map is None):
        raise Exception(f"No config provided.")
    if(cfg_map is not None):
        for k in cfg_map.keys():
            if(k not in self_obj.CONFIG_MAP):
                raise Exception(f"Unknown config map key: {k} from {self_obj.CONFIG_MAP.keys()}")

def setup_var_map(self_obj, var_map):
    if(var_map is not None):
        for k, v in var_map.items():
            name = k.split('.')
            parent = '.'.join(name[: -1])
            map_to = self_obj.VAR_MAP.get(parent)
            if(map_to is not None):
                obj = getattr(self_obj, map_to)
                setattr(obj, name[-1], v)
            #else:
                # ignore for multiple inheritance
                #raise Exception(f"Bad value: {parent}. Only possible: {self_obj.VAR_MAP.keys()}")

def setup_cfg_map(self_obj, args, cfg_map):
    if(cfg_map is not None):
        for k, cfg_class in self_obj.CONFIG_MAP.items():
            name = f'cfg_{k}' if 'cfg' not in k else 'cfg'
            if(k not in cfg_map):
                if(args is None): # setup if not set
                    setattr(self_obj, name, cfg_class())
            else: # override
                setattr(self_obj, name, cfg_map[k])

def setup_map(self_obj, args, cfg_map, var_map):
    setup_var_map(self_obj, var_map)
    setup_cfg_map(self_obj, args, cfg_map)