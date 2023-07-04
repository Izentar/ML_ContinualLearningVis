import torch
from utils.utils import search_kwargs

def default_reset_optim(lr, **kwargs):
    def inner_default_reset_optim(optim, optim_idx, sched=None):
        for g in optim.param_groups:
            g['lr'] = lr
    return inner_default_reset_optim

def exponential_scheduler(**kwargs):
    new_kwargs = search_kwargs(kwargs, ['gamma', 'last_epoch'])
    return lambda optim: torch.optim.lr_scheduler.ExponentialLR(optim, **new_kwargs)

def none(**kwargs):
    return None

def adam(**kwargs):
    new_kwargs = search_kwargs(kwargs, ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad'])
    return lambda param: torch.optim.Adam(param, **new_kwargs)

def adamw(**kwargs):
    new_kwargs = search_kwargs(kwargs, ['lr', 'betas', 'eps', 'weight_decay', 'amsgrad'])
    return lambda param: torch.optim.AdamW(param, **new_kwargs)

def sgd(**kwargs):
    new_kwargs = search_kwargs(kwargs, ['lr', 'momentum', 'dampening', 'weight_decay'])
    return lambda param: torch.optim.SGD(param, **new_kwargs)

def step_scheduler(**kwargs):
    new_kwargs = search_kwargs(kwargs, ['step_size', 'gamma', 'last_epoch'])
    return lambda optim: torch.optim.lr_scheduler.StepLR(optim, **new_kwargs)

def mulitstep_scheduler(**kwargs):
    new_kwargs = search_kwargs(kwargs, ['milestones', 'gamma', 'last_epoch'])
    return lambda optim: torch.optim.lr_scheduler.MultiStepLR(optim, **new_kwargs)

class ModelOptimizerManager():
    OPTIMIZERS = {
        'ADAM': adam,
        'ADAMW': adamw,
        'SGD': sgd,
    }

    SCHEDULERS = {
        'NONE': none,
        'EXPONENTIAL-SCHED': exponential_scheduler,
        'STEP-SCHED': step_scheduler,
        'MULTISTEP-SCHED': mulitstep_scheduler,
    }

    RESER_OPTIM = {
        'DEFAULT': default_reset_optim
    }

    def __init__(
        self,
        optimizer_type: str = None,
        scheduler_type: str = None,
        reset_optim_type: str = None,
    ) -> None:
        self.OPTIMIZERS = {k.upper(): v for k, v in self.OPTIMIZERS.items()}
        self.SCHEDULERS = {k.upper(): v for k, v in self.SCHEDULERS.items()}
        self.RESER_OPTIM = {k.upper(): v for k, v in self.RESER_OPTIM.items()}

        self.optimizer_type = optimizer_type
        self.scheduler_type = scheduler_type
        self.reset_optim_type = reset_optim_type

        if not (self.optimizer_type is None or isinstance(self.optimizer_type, str)):
            raise Exception(f'Bad optimizer type: {type(self.optimizer_type)}')
        if not (self.scheduler_type is None or isinstance(self.scheduler_type, str)):
            raise Exception(f'Bad scheduler type: {type(self.scheduler_type)}')
        if not (self.reset_optim_type is None or isinstance(self.reset_optim_type, str)):
            raise Exception(f'Bad reset optimizer type: {type(self.reset_optim_type)}')

    def get_optimizer(self, **kwargs):
        if(self.optimizer_type is None or self.optimizer_type.upper() == 'NONE'):
            return None
        return self.OPTIMIZERS[self.optimizer_type.upper()](**kwargs)

    def get_scheduler(self, **kwargs):
        if(self.scheduler_type is None or self.scheduler_type.upper() == 'NONE'):
            return None
        return self.SCHEDULERS[self.scheduler_type.upper()](**kwargs)

    def get_reset_optimizer_f(self, **kwargs):
        if(self.reset_optim_type is None or self.reset_optim_type.upper() == 'NONE'):
            return None
        return self.RESER_OPTIM[self.reset_optim_type.upper()](**kwargs)