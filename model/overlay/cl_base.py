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
from utils import pretty_print as pp
from typing import Any, Dict


class ClBase(LightningModule):
    @dataclass
    class Config(utils.BaseConfigDataclass):
        num_classes: int
        type: str # not used but required for now
        val_split: int

    @dataclass
    class Optimizer(utils.BaseConfigDataclass):        
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
    class Scheduler(utils.BaseConfigDataclass):
        type: str = None
        kwargs: dict = None

    def __init__(
        self, 
        data_passer:dict=None, 
        args=None,
        cfg_map=None,
        args_map=None,
    ):
        '''
            Optimizer.type - function with signature fun(parameters)
            Scheduler.type - function with signature fun(optim)
            Optimizer.reset_type - function with signature fun(optimizer)
        '''
        self.CONFIG_MAP, self.VAR_MAP = self._get_config_maps()
        utils.check_cfg_var_maps(self)

        super().__init__()
        self._map_cfg(args=args, cfg_map=cfg_map, args_map=args_map)
        # ignore *args, **kwargs

        self.optim_manager = ModelOptimizerManager(
            optimizer_type=self.cfg_optim.type,
            scheduler_type=self.cfg_sched.type,
            reset_optim_type=self.cfg_optim.reset_type,
        )

        assert not (self.cfg_sched.type is not None and self.cfg_optim.type is None), "Scheduler should have optimizer"

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self.train_acc_dream = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self._valid_accs = nn.ModuleDict()
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)

        self.data_passer = data_passer
        self.schedulers = None

        for i in range(self.cfg.val_split):
            self.valid_accs(i)

    def _get_config_maps(self):
        return {
            'cfg_optim': ClBase.Optimizer,
            'cfg_sched': ClBase.Scheduler,
            'cfg': ClBase.Config,
        },{
            'optim': 'cfg_optim',
            'sched': 'cfg_sched',
            '': 'cfg',
        }

    def _map_cfg(self, args, cfg_map:dict, args_map:dict):
        setup_args.check(self, args, cfg_map)

        if(args is not None):
            args = setup_args.setup_args(args_map, args, 'model')
            utils.setup_obj_dataclass_args(self, args=args, root_name='model', recursive=True, recursive_types=[Namespace])
        setup_args.setup_cfg_map(self, args, cfg_map)

    def valid_accs(self, idx):
        # OLD SOLUTION use metric.compute(), because self.log() did not registered this module :(
        # NEW SOLUTION predefine in __init__ all necessary metrics. If error occurs then 
        # the metric was not created inside __init__.
        try:
            return self._valid_accs[str(idx)]
        except KeyError:
            tmp = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes).to(self.device)
            self._valid_accs[str(idx)] = tmp
            return tmp

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        if "dream" not in batch:
            return self.training_step_normal(batch["normal"], optimizer_idx)
        if 'normal' not in batch:
            return self.training_step_dream(batch["dream"], optimizer_idx)
        
        new_batch_data = torch.cat((batch["normal"][0], batch["dream"][0]))
        new_batch_target = torch.cat((batch["normal"][1], batch["dream"][1]))

        loss = self.training_step_normal([new_batch_data, new_batch_target], optimizer_idx)
        return loss

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
    def training_step_normal(self, batch, optimizer_idx):
        pass

    @abstractmethod
    def training_step_dream(self, batch, optimizer_idx):
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

    @property
    def name(self):
        raise Exception("Not implemented")

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        pass

    def get_scheduler_construct(self, idx):
        schedulers_construct_f = self.optim_manager.get_scheduler(**self.cfg_sched.kwargs)
        if(idx != 0):
            raise Exception("Only one optimizer is present.")
        return schedulers_construct_f

    def _create_scheduler(self, optim, optim_idx):
        scheduler_construct_f = self.get_scheduler_construct(optim_idx)
        if(scheduler_construct_f is None):
            return 
        self.schedulers[optim_idx] = scheduler_construct_f(optim)
        self.print_sched_info(optim_idx, self.schedulers[optim_idx], scheduler_construct_f.kwargs)

    def _create_optimizer(self) -> torch.optim.Optimizer:
        optimizer_construct_f = self.optim_manager.get_optimizer(**self.cfg_optim.kwargs)

        if(optimizer_construct_f is None):
            optim = torch.optim.Adam(self.parameters(), lr=optim_Adam_config["lr"])
        else:
            optim = optimizer_construct_f(self.parameters())     
        return optim

    def configure_optimizers(self):
        optims = self._create_optimizer()
        self.schedulers = {}
        if(not isinstance(optims, Sequence)):
            self.print_optim_info(0, optims)
            self._create_scheduler(optims, optim_idx=0)
        else:
            for optim_idx, optim in enumerate(optims):
                self.print_optim_info(optim_idx, optim)
                self._create_scheduler(optim, optim_idx=optim_idx)
        return optims

    # training_epoch_end
    def training_epoch_end(self, output):
        optims = self.optimizers()
        if(not isinstance(optims, Sequence)):
            optims = [optims]
        for optim_idx, opt in enumerate(optims):
            if(optim_idx not in self.schedulers):
                continue
            if(self.current_epoch >= self.data_passer['epoch_per_task'] - 1):
                self._training_epoch_end_last_epoch(opt=opt, optim_idx=optim_idx)
            else: # if it is not the last epoch
                self._training_epoch_end_call_sched(opt=opt, optim_idx=optim_idx)
                
    def _training_epoch_end_last_epoch(self, opt, optim_idx):
        # invoke after the last epoch in training, when you switch to another task
        self.optim_manager.get_reset_optimizer_f(**self.cfg_optim.kwargs)(opt, optim_idx=optim_idx)
        pp.sprint(f'{pp.COLOR.NORMAL}Scheduler restarted at epoch {self.current_epoch} end. Learning rate: {self.schedulers[optim_idx].get_last_lr()}')
        self._create_scheduler(opt, optim_idx=optim_idx)

    def _training_epoch_end_call_sched(self, opt, optim_idx):
        # if it is not the last epoch, call scheduler
        last_lr = self.schedulers[optim_idx].get_last_lr()
        self.schedulers[optim_idx].step()
        if(last_lr != self.schedulers[optim_idx].get_last_lr()):
            pp.sprint(f"{pp.COLOR.NORMAL}Changed learning rate from: '{last_lr}' to: '{self.schedulers[optim_idx]._last_lr}' for optimizer index: {optim_idx} at epoch: {self.current_epoch}")

    def print_optim_info(self, idx, optim):
        pp.sprint(f"{pp.COLOR.NORMAL}INFO: Created {pp.COLOR.NORMAL_4}{idx}{pp.COLOR.NORMAL} optim config: {pp.COLOR.NORMAL_3}{optim}")

    def print_sched_info(self, idx, sched_class, sched_info):
        pp.sprint(f"{pp.COLOR.NORMAL}INFO: Created {pp.COLOR.NORMAL_4}{idx}{pp.COLOR.NORMAL} sched config: {pp.COLOR.NORMAL_3}{type(sched_class).__name__}\n{sched_info}")