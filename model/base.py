from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.autograd.variable import Variable
import torchmetrics

from config.default import optim_Adam_config
from typing import Union, Sequence
from abc import abstractmethod
from utils.functional.model_optimizer import ModelOptimizerManager

from dataclasses import dataclass, field
from argparse import Namespace
from utils import utils, setup_args


class CLBase(LightningModule):
    @dataclass
    class Config():
        num_classes: int
        type: str # not used but required for now

    @dataclass
    class Optimizer():        
        type: str = 'adam'
        reset_type: str = None
        kwargs: dict = None

        def __post_init__(self):
            if self.kwargs is None:
                self.kwargs = {
                    'lr': 1e-3,
                    'gamma': 1.,
                }

    @dataclass
    class Scheduler():
        type: str = None
        steps: Sequence[int] = None
        kwargs: dict = None

        def __post_init__(self):
            self.steps = self.steps if self.steps is not None else (None, )
            if self.kwargs is None:
                self.kwargs = {
                    'lr': 1e-3,
                    'gamma': 1.,
                }

    def __init__(
        self, 
        data_passer:dict=None, 
        args=None,
        cfg_map=None,
        var_map=None,
        #*aargs,
        #**akwargs,
    ):
        '''
            Optimizer.type - function with signature fun(parameters)
            Scheduler.type - function with signature fun(optim)
            Optimizer.reset_type - function with signature fun(optimizer)
        '''
        self.CONFIG_MAP, self.VAR_MAP = self._get_config_maps()

        super().__init__()
        self._map_cfg(args=args, cfg_map=cfg_map, var_map=var_map)
        # ignore *args, **kwargs

        self.optim_manager = ModelOptimizerManager(
            optimizer_type=self.cfg_optim.type,
            scheduler_type=self.cfg_sched.type,
            reset_optim_type=self.cfg_optim.reset_type,
        )

        assert not (self.cfg_sched.type is not None and self.cfg_optim.type is None), "Scheduler should have optimizer"
        if(self.cfg_sched.steps is not None and self.cfg_sched.type is None):
            print('WARNIGN: Using "cfg_sched.steps" but scheduler is not selected.')

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self.train_acc_dream = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self._valid_accs = nn.ModuleDict()
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self.num_classes = self.cfg.num_classes

        self.data_passer = data_passer
        self.optimizer_construct_f = self.optim_manager.get_optimizer(**self.cfg_optim.kwargs)
        self.scheduler_construct_f = self.optim_manager.get_scheduler(**self.cfg_sched.kwargs)
        self.optimizer_restart_params_f = self.optim_manager.get_reset_optimizer_f(**self.cfg_optim.kwargs)

        self.scheduler = None
        self.optimizer = None

    def _get_config_maps(self):
        return {
            'optim': CLBase.Optimizer,
            'sched': CLBase.Scheduler,
            'cfg': CLBase.Config,
        },{
            'optimizer': 'cfg_optim',
            'scheduler': 'cfg_sched',
            'config': 'cfg',
        }
    
    def _map_from_args(self, args, not_from):
        if(args is not None):
            self.cfg = self.CONFIG_MAP['cfg'](
                **utils.get_obj_dict_dataclass(args.model, not_from, self.CONFIG_MAP['cfg'])
            )
            self.cfg_optim=self.CONFIG_MAP['optim'](
                **utils.get_obj_dict(args.model.optim, recursive=True, recursive_types=[Namespace])
            )
            self.cfg_sched=self.CONFIG_MAP['sched'](
                **utils.get_obj_dict(args.model.scheduler, recursive=True, recursive_types=[Namespace])
            )

    def _map_cfg(self, args, cfg_map:dict, var_map:dict):
        setup_args.check(self, args, cfg_map)
        not_from = Namespace
        if(args is not None):
            self._map_from_args(args, not_from)
        setup_args.setup_map(self, args, cfg_map, var_map)

    def valid_accs(self, idx):
        # use metric.compute(), because self.log() did not registered this module :(
        try:
            return self._valid_accs[str(idx)]
        except KeyError:
            tmp = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes).to(self.device)
            self._valid_accs[str(idx)] = tmp
            return tmp

    def training_step(self, batch, batch_idx):
        if "dream" not in batch:
            return self.training_step_normal(batch["normal"])
        if 'normal' not in batch:
            return self.training_step_dream(batch["dream"])

        loss_normal = self.training_step_normal(batch["normal"])
        loss_dream = self.training_step_dream(batch["dream"])
        return loss_normal + loss_dream

    def get_model_out_data(self, model_out):
        """
            Return latent and model output dictionary if exist else None
        """
        model_out_dict = None
        latent = model_out
        if(isinstance(model_out, tuple)):
            latent = model_out[0]
            model_out_dict = model_out[1]
        return latent, model_out_dict

    @abstractmethod
    def training_step_normal(self, batch):
        pass

    @abstractmethod
    def training_step_dream(self, batch):
        pass

    @abstractmethod
    def call_loss(self, input, target):
        pass

    @abstractmethod
    def get_objective_target_name(self):
        """
            It should return the target layer name currently used in model.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def get_root_objective_target(self): 
        """
            Return root string with a hard space at the end.
        """
        pass

    @abstractmethod
    def get_obj_str_type(self) -> str:
        pass

    @abstractmethod
    def get_objective_layer(self):
        pass

    @abstractmethod
    def get_objective_layer_output_shape(self):
        pass

    @abstractmethod
    def init_weights(self):
        pass

    def configure_optimizers(self):
        if(self.optimizer_construct_f is None):
            optim = torch.optim.Adam(self.parameters(), lr=optim_Adam_config["lr"])
        else:
            optim = self.optimizer_construct_f(self.parameters())
        
        if(self.scheduler_construct_f is not None):
            self.scheduler = self.scheduler_construct_f(optim)
        self.optimizer = optim
        return optim

    # training_epoch_end
    def training_epoch_end(self, output):
        # OK
        #print(f"debug current_task_loop: {self.data_passer['current_task_loop']}, self.current_epoch {self.current_epoch}")
        #print(f"self.data_passer['epoch_per_task'] {self.data_passer['epoch_per_task']}, self.cfg_sched.steps {self.cfg_sched.steps}")
        if(self.scheduler is not None):
            if(self.current_epoch >= self.data_passer['epoch_per_task'] - 1):
                self.optimizer_restart_params_f(self.optimizer)
                self.scheduler = self.scheduler_construct_f(self.optimizer)
                print(f'Scheduler restarted at epoch {self.current_epoch} end. Learning rate: {self.scheduler._last_lr}')
                return
            if(self.current_epoch in self.cfg_sched.steps):
                self.scheduler.step()
                print(f"Changed learning rate to: {self.scheduler._last_lr}")
                #print(f"debug current_task_loop: {self.data_passer['current_task_loop']}, self.cfg_sched.steps {self.cfg_sched.steps}")
