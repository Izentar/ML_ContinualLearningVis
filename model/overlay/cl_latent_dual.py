from typing import Any, Dict
import torch
from config.default import optim_Adam_config
from loss_function.chiLoss import DummyLoss
from utils import pretty_print as pp
from torch.nn.functional import relu
import torchmetrics
from copy import copy

from dataclasses import dataclass
from model.overlay.cl_latent_chi import ClLatentChi
from utils.functional.model_optimizer import ModelOptimizerManager
from collections.abc import Sequence

class ModelSufix(torch.nn.Module):
    def __init__(self, model, num_classes) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes
        self.vis_inner = False

        self.ln = torch.nn.Linear(self.model.get_objective_layer().out_features, self.model.get_objective_layer().out_features)
        self.ln2 = torch.nn.Linear(self.model.get_objective_layer().out_features, self.model.get_objective_layer().out_features)
        self.ln3 = torch.nn.Linear(self.model.get_objective_layer().out_features, num_classes)    
    
    def forward(self, x, **kwargs):
        xe = relu(self.model(x))
        xe = relu(self.ln(xe))
        xe = relu(self.ln2(xe))
        xe = self.ln3(xe)
        return xe
    
    def set_visualization_type(self, vis_inner):
        self.vis_inner = vis_inner

    def get_objective_layer_name(self):
        if(self.vis_inner):
            return "model." + self.model.get_objective_layer_name()
        return "ln3"

    def get_root_name(self):
        if(self.vis_inner):
            return "model." + self.model.get_root_name()
        return ""

    def get_objective_layer(self):
        if(self.vis_inner):
            return self.model.get_objective_layer()
        return self.ln3

    def get_objective_layer_output_shape(self):
        if(self.vis_inner):
            return self.model.get_objective_layer_output_shape()
        return (self.ln3.out_features,)
    
    def outer_params(self):
        return [self.ln.weight, self.ln2.weight, self.ln3.weight]
    
    @property
    def name(self):
        return self.model.name

class ClLatentDual(ClLatentChi):
    @dataclass
    class Loss():
        @dataclass
        class Chi():
            @dataclass
            class Dual():
                inner_scale: float = 1.
                outer_scale: float = 1.

                def __post_init__(self):
                    if not (0. <= self.inner_scale and self.inner_scale <= 1.):
                        raise Exception(f"Inner scale param can be only in range of [0, 1]")
                    if not (0. <= self.outer_scale and self.outer_scale <= 1.):
                        raise Exception(f"Outer scale param can be only in range of [0, 1]")

    @dataclass
    class Outer():
        @dataclass
        class Optimizer():        
            type: str = None
            reset_type: str = None
            kwargs: dict = None

        @dataclass
        class Scheduler():
            type: str = None
            kwargs: dict = None

    @dataclass
    class Inner():
        @dataclass
        class Config():
            visualize_type: str = 'outer'

    def __after_init_sched__(source_sched, other_sched):
        if source_sched.kwargs is not None:
            for k, v in source_sched.kwargs.items():
                if v is None:
                    source_sched.kwargs[k] = other_sched.kwargs[k]
        else:
            source_sched.kwargs = copy(other_sched.kwargs)
        if source_sched.type is None:
            source_sched.type = other_sched.type
                
    def __after_init_optim__(source_optim, other_optim):
        if source_optim.kwargs is not None:
            for k, v in source_optim.kwargs.items():
                if v is None:
                    source_optim.kwargs[k] = other_optim.kwargs[k]
        else:
            source_optim.kwargs = copy(other_optim.kwargs)
        if source_optim.type is None:
            source_optim.type = other_optim.type
        if source_optim.reset_type is None:
            source_optim.reset_type = other_optim.reset_type

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg_loss_chi_dual': ClLatentDual.Loss.Chi.Dual,
            'cfg_outer_optim': ClLatentDual.Outer.Optimizer,
            'cfg_outer_sched': ClLatentDual.Outer.Scheduler,
            'cfg_inner_cfg': ClLatentDual.Inner.Config,
        })
        b.update({
            'loss.chi.dual': 'cfg_loss_chi_dual',
            'outer.optim': 'cfg_outer_optim',
            'outer.sched': 'cfg_outer_sched',
            'inner.cfg': 'cfg_inner_cfg',
        })
        return a, b

    def __init__(
            self, 
            *args,
            model:torch.nn.Module=None,
            **kwargs
        ):
        num_classes = kwargs['args'].model.num_classes
        model = ModelSufix(model, num_classes)
        super().__init__(*args, model=model, **kwargs)
        ClLatentDual.__after_init_optim__(self.cfg_outer_optim, self.cfg_optim)
        ClLatentDual.__after_init_sched__(self.cfg_outer_sched, self.cfg_sched)
        self._outer_loss_f = DummyLoss(torch.nn.CrossEntropyLoss())

        self._hook_handle = model.model.get_objective_layer().register_forward_hook(self._hook)

        self.valid_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes).to(self.device)
        self.test_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self.train_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self._saved_loss = 0.0
        

        self.outer_optim_manager = ModelOptimizerManager(
            optimizer_type=self.cfg_outer_optim.type,
            scheduler_type=self.cfg_outer_sched.type,
            reset_optim_type=self.cfg_outer_optim.reset_type,
        )

        if(self.cfg_inner_cfg.visualize_type == 'inner'):
            model.set_visualization_type(True)
        elif(self.cfg_inner_cfg.visualize_type == 'outer'):
            model.set_visualization_type(False)  
        else:
            raise Exception(f"Bad value: {self.cfg_inner_cfg.visualize_type}")
        
        if(self.cfg_loss_chi_dual.inner_scale == 0.):
            pp.sprint(f"{pp.COLOR.NORMAL}INFO: {self.name} inner scale set to zero.")
        if(self.cfg_loss_chi_dual.outer_scale == 0.):
            pp.sprint(f"{pp.COLOR.NORMAL}INFO: {self.name} outer scale set to zero.")
        if(self.cfg_loss_chi_dual.inner_scale == 0. and self.cfg_loss_chi_dual.outer_scale == 0.):
            raise Exception("Both losses (inner, outer) cannot be zero!")

        # Not need here to enable this. Calculating double loss in this case does not
        # affect the result. It can only make training slower. Here I use optimizer_idx argument. 
        #self.automatic_optimization = False

    def _hook(self, module, input, output):
        self._inner_output = output

    def _is_inner_turn(self, optimizer_idx=None):
        return (self._outer_optim_disabled or optimizer_idx is None) or (not self._inner_optim_disabled and optimizer_idx == 0)

    def _is_inner_enabled(self, optimizer_idx=None):
        return (self._is_inner_turn(optimizer_idx) and self.cfg_loss_chi_dual.inner_scale != 0.) or self.cfg_loss_chi_dual.outer_scale == 0.

    def _is_outer_turn(self, optimizer_idx=None):
        return optimizer_idx is None or (not self._outer_optim_disabled and optimizer_idx == 1)

    def _is_outer_enabled(self, optimizer_idx=None):
        return (self._is_outer_turn(optimizer_idx) and self.cfg_loss_chi_dual.outer_scale != 0)

    def process_losses_normal(self, x, y, latent, log_label, optimizer_idx, model_out_dict=None):
        if(self._is_inner_enabled(optimizer_idx)):
            loss_inner = self._loss_f(self._inner_output, y) * self.cfg_loss_chi_dual.inner_scale
            self.log(f"{log_label}/island_CHI-K", loss_inner)
            return loss_inner
        
        if(self._is_outer_enabled(optimizer_idx)):
            loss_outer = self._outer_loss_f(latent, y) * self.cfg_loss_chi_dual.outer_scale
            self.log(f"{log_label}/island_CROSS-E", loss_outer)
            return loss_outer
        raise Exception("Both losses (inner, outer) cannot be zero!")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        optim2_params = None
        optimizer_construct_f = self.optim_manager.get_optimizer(**self.cfg_optim.kwargs)
        if(self.cfg_loss_chi_dual.outer_scale == 0.):
            self._outer_optim_disabled = True
        else:
            self._outer_optim_disabled = False  
        if(self.cfg_loss_chi_dual.inner_scale == 0.):
            self._inner_optim_disabled = True
        else:
            self._inner_optim_disabled = False  
            
        if(self.cfg_loss_chi_dual.inner_scale == 0. or self.cfg_loss_chi_dual.outer_scale == 0.):
            optim_params = self.model.parameters() # all params even if we will calc gradient for partial loss
        else:
            optim_params = self.model.model.parameters()
            optim2_params = self.model.outer_params()

        if(optimizer_construct_f is None):
            optim_construct_f = lambda params: torch.optim.Adam(params, lr=optim_Adam_config["lr"])
        else:
            optim_construct_f = lambda params: optimizer_construct_f(params) 

        optimizer_construct_outer_f = self.outer_optim_manager.get_optimizer(**self.cfg_outer_optim.kwargs)
        if(optimizer_construct_outer_f is None):
            raise Exception("Internal error, optimizer_construct_outer_f cannot be None")
        optim2_construct_f = lambda params: optimizer_construct_outer_f(params)  

        optim = optim_construct_f(optim_params)
        if(optim2_params is None):
            pp.sprint(f"{pp.COLOR.NORMAL}INFO: Used {pp.COLOR.NORMAL_4}optim{pp.COLOR.NORMAL} config: {self.cfg_optim}")
            return optim
            
        optim2 = optim2_construct_f(optim2_params)
        pp.sprint(f"{pp.COLOR.NORMAL}INFO: Used {pp.COLOR.NORMAL_4}outer optim{pp.COLOR.NORMAL} config: {self.cfg_outer_optim}")
        return optim, optim2
    
    def get_scheduler_construct(self, idx):
        match idx:
            case 0:
                return super().get_scheduler_construct(idx)
            case 1:
                pp.sprint(f"{pp.COLOR.NORMAL_2}INFO: Used {pp.COLOR.NORMAL_4}outer sched{pp.COLOR.NORMAL_2} config: {self.cfg_outer_sched}")
                return self.outer_optim_manager.get_scheduler(**self.cfg_outer_sched.kwargs)
            case _:
                raise Exception(f"Expected at most 2 optimizers. Requested index: {idx}")
    
    def _validation_step_inner(self, y):
        if(not self._is_inner_enabled()):
            return
        val_loss_inner = self._loss_f(self._inner_output, y, train=False)
        self.log("val_last_step_loss_inner", val_loss_inner, on_epoch=True)

        self.valid_acc_inner(self._loss_f.classify(self._inner_output), y)
        self.log("valid_acc_inner", self.valid_acc_inner.compute())

    def _validation_step_outer(self, latent, y, dataloader_idx):
        if(not self._is_outer_enabled()):
            return
        val_loss_outer = self._outer_loss_f(latent, y, train=False)
        self.log("val_last_step_loss_outer", val_loss_outer, on_epoch=True)
        
        valid_acc = self.valid_accs(dataloader_idx)
        valid_acc(self._outer_loss_f.classify(latent), y)
        self.log("valid_acc_outer", valid_acc.compute())

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        self._validation_step_inner(y=y)
        self._validation_step_outer(latent=latent, y=y, dataloader_idx=dataloader_idx)

    def _test_step_inner(self, y): 
        if(not self._is_inner_enabled()):
            return
        test_loss_inner = self._loss_f(self._inner_output, y, train=False)
        self.log("test_loss_inner", test_loss_inner)

        #print(self._loss_f.classify(self._inner_output), y)
        self.test_acc_inner(self._loss_f.classify(self._inner_output), y)
        self.log("test_acc_inner", self.test_acc_inner)

    def _test_step_outer(self, latent, y):
        if(not self._is_outer_enabled()):
            return
        test_loss_outer = self._outer_loss_f(latent, y, train=False)
        self.log("test_loss_outer", test_loss_outer)

        self.test_acc(self._outer_loss_f.classify(latent), y)
        self.log("test_acc_outer", self.test_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        self._test_step_inner(y)
        self._test_step_outer(latent=latent, y=y)

        self._test_step_log_data()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        loss = super().training_step(batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx)
        self._saved_loss = 0.0
        if(self.cfg_loss_chi_dual.inner_scale != 0. and self.cfg_loss_chi_dual.outer_scale != 0.):
            self._saved_loss = loss.item()
        return loss
    
    def training_step_acc_inner(self, x, y, latent):
        self.train_acc_inner(self._loss_f.classify(self._inner_output), y)
        self.log("train_step_acc_inner", self.train_acc_inner, on_step=False, on_epoch=True) 
        
    def training_step_acc_outer(self, x, y, latent):
        self.train_acc(self._outer_loss_f.classify(latent), y)
        self.log("train_step_acc_outer", self.train_acc, on_step=False, on_epoch=True) 
    
    def training_step_acc(self, x, y, loss, latent, model_out_dict, optimizer_idx):
        self.log("train_loss/total", loss.item() + self._saved_loss)

        if(self._is_inner_enabled(optimizer_idx)):
            self.training_step_acc_inner(x=x, y=y, latent=latent)
            return loss
        if(self._is_outer_enabled(optimizer_idx)):
            self.training_step_acc_outer(x=x, y=y, latent=latent)
            return loss
        
        raise Exception('Unacceptable state.')

    def get_obj_str_type(self) -> str:
        return 'ClLatentDual_' + super().get_obj_str_type()
        
class ClLatentDualHalved(ClLatentDual):

    @dataclass
    class Inner():
        @dataclass
        class Config(ClLatentDual.Inner.Config):
            partial_backward: bool = True

        @dataclass
        class First():
            @dataclass
            class Optimizer():        
                type: str = None
                reset_type: str = None
                kwargs: dict = None

            @dataclass
            class Scheduler():
                type: str = None
                kwargs: dict = None

        @dataclass
        class Second():
            @dataclass
            class Optimizer():        
                type: str = None
                reset_type: str = None
                kwargs: dict = None
            
            @dataclass
            class Scheduler():
                type: str = None
                kwargs: dict = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

        ClLatentDual.__after_init_optim__(self.cfg_inner_first_optim, self.cfg_optim)
        ClLatentDual.__after_init_optim__(self.cfg_inner_second_optim, self.cfg_optim)

        ClLatentDual.__after_init_sched__(self.cfg_inner_first_sched, self.cfg_sched)
        ClLatentDual.__after_init_sched__(self.cfg_inner_second_sched, self.cfg_sched)


        self.inner_first_half_optim_manager = ModelOptimizerManager(
            optimizer_type=self.cfg_inner_first_optim.type,
            scheduler_type=self.cfg_inner_first_sched.type,
        )
        self.inner_second_half_optim_manager = ModelOptimizerManager(
            optimizer_type=self.cfg_inner_second_optim.type,
            scheduler_type=self.cfg_inner_second_sched.type,
        )

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg_inner_first_optim': ClLatentDualHalved.Inner.First.Optimizer,
            'cfg_inner_second_optim': ClLatentDualHalved.Inner.Second.Optimizer,
            'cfg_inner_first_sched': ClLatentDualHalved.Inner.First.Scheduler,
            'cfg_inner_second_sched': ClLatentDualHalved.Inner.Second.Scheduler,
            'cfg_inner_cfg': ClLatentDualHalved.Inner.Config,
        })
        b.update({
            'inner.first.optim': 'cfg_inner_first_optim',
            'inner.second.optim': 'cfg_inner_second_optim',
            'inner.first.sched': 'cfg_inner_first_sched',
            'inner.second.sched': 'cfg_inner_second_sched',
            'inner.cfg': 'cfg_inner_cfg',
        })
        return a, b
    
    def get_scheduler_construct(self, idx):
        match idx:
            case 0:
                pp.sprint(f"{pp.COLOR.NORMAL_2}INFO: Used {pp.COLOR.NORMAL_4}inner first sched{pp.COLOR.NORMAL_2} config: {self.cfg_inner_first_sched}")
                return self.inner_first_half_optim_manager.get_scheduler(**self.cfg_inner_first_sched.kwargs)
            case 1:
                pp.sprint(f"{pp.COLOR.NORMAL_2}INFO: Used {pp.COLOR.NORMAL_4}inner second sched{pp.COLOR.NORMAL_2} config: {self.cfg_inner_second_sched}")
                return self.inner_second_half_optim_manager.get_scheduler(**self.cfg_inner_second_sched.kwargs)
            case 2:
                pp.sprint(f"{pp.COLOR.NORMAL_2}INFO: Used {pp.COLOR.NORMAL_4}outer sched{pp.COLOR.NORMAL_2} config: {self.cfg_outer_sched}")
                return self.outer_optim_manager.get_scheduler(**self.cfg_outer_sched.kwargs)
            case _:
                raise Exception(f"Expected at most 2 optimizers. Requested index: {idx}")

    def _create_optimizer(self) -> torch.optim.Optimizer:
        optims = super()._create_optimizer()
        halves = self._create_optimizer_halves()
        if(self._outer_optim_disabled):
            return halves
        if(isinstance(optims, Sequence)):
            halves.append(optims[-1])
            return halves
        halves.append(optims)
        return halves 
            
    def _create_optimizer_halves(self):
        optim_1 = self.inner_first_half_optim_manager.get_optimizer(**self.cfg_inner_first_optim.kwargs)
        optim_2 = self.inner_second_half_optim_manager.get_optimizer(**self.cfg_inner_second_optim.kwargs)
        if(optim_1 is None):
            optim_first_half = torch.optim.Adam(self.model.model.first_half_params(), lr=optim_Adam_config["lr"])
        else:
            optim_first_half = optim_1(self.model.model.first_half_params())

        if(optim_2 is None):
            optim_second_half = torch.optim.Adam(self.model.model.second_half_params(), lr=optim_Adam_config["lr"])
        else:
            optim_second_half = optim_2(self.model.model.second_half_params())
        pp.sprint(f"{pp.COLOR.NORMAL}INFO: Used {pp.COLOR.NORMAL_4}inner first optim{pp.COLOR.NORMAL} config: {self.cfg_inner_first_optim}")
        pp.sprint(f"{pp.COLOR.NORMAL}INFO: Used {pp.COLOR.NORMAL_4}inner second optim{pp.COLOR.NORMAL} config: {self.cfg_inner_second_optim}")
        return [optim_first_half, optim_second_half]
    
    def training_step_acc(self, x, y, loss, latent, model_out_dict, optimizer_idx):
        self.log("train_loss/total", loss)
        self.train_acc(self._loss_f.classify(latent), y)
        self.log("train_step_acc", self.train_acc, on_step=False, on_epoch=True)

    def training_step(self, batch, batch_idx):
        # call grandparent method
        super(ClLatentDual, self).training_step(batch=batch, batch_idx=batch_idx, optimizer_idx=None)
        # do not return anything, only change optimizer_idx to None

    def _process_losses_normal_full_backward(self, y, latent, log_label):
        loss_inner_item = 0.
        loss_outer_item = 0.
        sum_loss = None

        if(self._is_outer_enabled()):
            loss_outer = self._outer_loss_f(latent, y) * self.cfg_loss_chi_dual.outer_scale
            self.log(f"{log_label}/island_CROSS-E", loss_outer)
            loss_outer_item = loss_outer.item()
            sum_loss = loss_outer

        if(self._is_inner_enabled()):
            loss_inner = self._loss_f(self._inner_output, y) * self.cfg_loss_chi_dual.inner_scale
            self.log(f"{log_label}/island_CHI-K", loss_inner)
            loss_inner_item = loss_inner.item()
            if(sum_loss is None):
                sum_loss = loss_inner
            else:
                sum_loss += loss_inner
        if(sum_loss is not None):
            self.manual_backward(sum_loss)
            
        return loss_inner_item, loss_outer_item
    
    def _process_losses_normal_partial_backward(self, y, latent, log_label):
        loss_inner_item = 0.
        loss_outer_item = 0.
        backward = False

        if(self._is_outer_enabled()):
            loss_outer = self._outer_loss_f(latent, y) * self.cfg_loss_chi_dual.outer_scale
            self.log(f"{log_label}/island_CROSS-E", loss_outer)
            loss_outer_item = loss_outer.item()

            self.manual_backward(loss_outer)
            backward = True # backward only once from this point

        if(self._is_inner_enabled()):
            loss_inner = self._loss_f(self._inner_output, y) * self.cfg_loss_chi_dual.inner_scale
            self.log(f"{log_label}/island_CHI-K", loss_inner)
            loss_inner_item = loss_inner.item()

            if not backward:
                self.manual_backward(loss_inner)

        return loss_inner_item, loss_outer_item

    def process_losses_normal(self, x, y, latent, log_label, optimizer_idx):
        outer_optim = None
        optims = self.optimizers()
        if(len(optims) == 2):
            first_half_optim, second_half_optim = optims
        elif(len(optims) == 3):
            first_half_optim, second_half_optim, outer_optim = optims
            outer_optim.zero_grad()
        else:
            raise Exception('Invalid internal state.')
        
        first_half_optim.zero_grad()
        second_half_optim.zero_grad()

        if(self.cfg_inner_cfg.partial_backward):
            ret = self._process_losses_normal_partial_backward(y=y, latent=latent, log_label=log_label)
        else:
            ret = self._process_losses_normal_full_backward(y=y, latent=latent, log_label=log_label)
            
        if(outer_optim is not None and self._is_outer_enabled()):
            outer_optim.step()
        second_half_optim.step()
        first_half_optim.step()
        return ret
        
        

    def training_step_normal(self, batch, optimizer_idx):
        x, y = batch
        latent, _ = self.training_step_normal_setup(x=x, y=y)
        log_label = "train_loss"

        loss_inner_item, loss_outer_item = self.process_losses_normal(
            x=x, 
            y=y, 
            latent=latent, 
            log_label=log_label,
            optimizer_idx=optimizer_idx,
        )
        
        self.log(f"{log_label}/total", loss_inner_item + loss_outer_item)

        if(self._is_inner_enabled()):
            self.training_step_acc_inner(x=x, y=y, latent=latent)
        if(self._is_outer_enabled()):
            self.training_step_acc_outer(x=x, y=y, latent=latent)
        return 0.0
    
    def training_step_dream(self, batch, optimizer_idx):
        x, y = batch
        latent, model_out_dict = self.training_step_dream_setup(x=x, y=y)
        log_label = "train_loss_dream"

        loss_inner_item, loss_outer_item = self.process_losses_normal(
            x=x, 
            y=y, 
            latent=latent,  
            log_label=log_label,
            optimizer_idx=optimizer_idx,
        )
        self.log(f"{log_label}/total", loss_inner_item + loss_outer_item)
        self.train_acc_dream(self._loss_f.classify(latent), y)
        self.log(f"{log_label}train_step_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)

        return 0.0