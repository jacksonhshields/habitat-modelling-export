

def get_dict_arguments(cfg, rm_keys=['type']):
    if rm_keys is None:
        rm_keys = ['type']
    args = {}
    for k,v in cfg.items():
        if k not in rm_keys:
            args[k] = v
    return args
