import torch

def _search_kwargs(kwargs: dict, vals:list[str]):
    new_kwargs = {}
    for k, v in kwargs.items():
        if(k in vals):
            new_kwargs[k] = v
    return new_kwargs

def default_reset_optim_init(lr, **kwargs):
    def default_reset_optim(optim, sched=None):
        for g in optim.param_groups:
            g['lr'] = lr

def exponential_lr(**kwargs):
    new_kwargs = _search_kwargs(kwargs, ['gamma', 'last_epoch'])
    return lambda optim: torch.optim.lr_scheduler.ExponentialLR(optim, new_kwargs)

def none(**kwargs):
    return None

def adam(**kwargs):
    new_kwargs = _search_kwargs(kwargs, ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad'])
    return lambda param: torch.optim.Adam(param, **new_kwargs)


class ModelOptimizerManager():
    OPTIMIZERS = {
        'ADAM': adam
    }

    SCHEDULERS = {
        'NONE': none,
        'EXPONENTIAL-LR': exponential_lr
    }

    RESER_OPTIM = {
        'DEFAULT': default_reset_optim_init
    }

    def __init__(
        self,
        optimizer_type:str,
        scheduler_type:str,
        reset_optim_type:str,
    ) -> None:
        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.reset_optim_type = reset_optim_type

    def get_optimizer(self, **kwargs):
        if(self.optimizer_type.upper() == 'NONE'):
            return None
        return self.OPTIMIZERS[self.optimizer_type.upper()](**kwargs)

    def get_scheduler(self, **kwargs):
        if(self.scheduler_type.upper() == 'NONE'):
            return None
        return self.SCHEDULERS[self.scheduler_type.upper()](**kwargs)

    def get_reset_optimizer_f(self, **kwargs):
        if(self.reset_optim_type.upper() == 'NONE'):
            return None
        return self.RESER_OPTIM[self.reset_optim_type.upper()](**kwargs)