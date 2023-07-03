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

class ModelSufix(torch.nn.Module):
    def __init__(self, model, num_classes) -> None:
        super().__init__()
        self.model = model
        self.num_classes = num_classes

        self.ln = torch.nn.Linear(self.model.get_objective_layer().out_features, self.model.get_objective_layer().out_features)
        self.ln2 = torch.nn.Linear(self.model.get_objective_layer().out_features, self.model.get_objective_layer().out_features)
        self.ln3 = torch.nn.Linear(self.model.get_objective_layer().out_features, num_classes)    
    
    def forward(self, x, **kwargs):
        xe = relu(self.model(x))
        xe = relu(self.ln(xe))
        xe = relu(self.ln2(xe))
        xe = self.ln3(xe)
        return xe

    def get_objective_layer_name(self):
        return "ln3"

    def get_root_name(self):
        return ""

    def get_objective_layer(self):
        return self.ln3

    def get_objective_layer_output_shape(self):
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
                
    def __after_init__(source_optim, other_optim):
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
        })
        b.update({
            'loss.chi.dual': 'cfg_loss_chi_dual',
            'outer.optim': 'cfg_outer_optim',
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
        ClLatentDual.__after_init__(self.cfg_outer_optim, self.cfg_optim)
        self._outer_loss_f = DummyLoss(torch.nn.CrossEntropyLoss())

        self._hook_handle = model.model.get_objective_layer().register_forward_hook(self._hook)

        self.valid_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes).to(self.device)
        self.test_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self.train_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self._saved_loss = 0.0

        outer_optim_manager = ModelOptimizerManager(
            optimizer_type=self.cfg_outer_optim.type,
            reset_optim_type=self.cfg_outer_optim.reset_type,
        )
        self.optimizer_construct_outer_f = outer_optim_manager.get_optimizer(**self.cfg_outer_optim.kwargs)
        
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
        return (not self._dual_optim or optimizer_idx is None) or (self._dual_optim and optimizer_idx == 0)

    def _is_inner_enabled(self, optimizer_idx=None):
        return (self._is_inner_turn(optimizer_idx) and self.cfg_loss_chi_dual.inner_scale != 0.) or self.cfg_loss_chi_dual.outer_scale == 0.

    def _is_outer_turn(self, optimizer_idx=None):
        return (not self._dual_optim or optimizer_idx is None) or (self._dual_optim and optimizer_idx == 1)

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
        optim2 = None
        if(self.optimizer_construct_f is None):
            if(self.cfg_loss_chi_dual.inner_scale == 0. or self.cfg_loss_chi_dual.outer_scale == 0.):
                optim = torch.optim.Adam(self.model.parameters(), lr=optim_Adam_config["lr"])
                self._dual_optim = False
            else:
                optim = torch.optim.Adam(self.model.model.parameters(), lr=optim_Adam_config["lr"])
                optim2 = torch.optim.Adam(self.model.outer_params(), lr=optim_Adam_config["lr"])
                self._dual_optim = True  
        else:
            if(self.cfg_loss_chi_dual.inner_scale == 0. or self.cfg_loss_chi_dual.outer_scale == 0.):
                optim = self.optimizer_construct_f(self.model.parameters()) 
                self._dual_optim = False
            else:
                assert self.optimizer_construct_outer_f is not None, "Internal error, optimizer_construct_outer_f cannot be None"
                optim = self.optimizer_construct_f(self.model.model.parameters())        
                optim2 = self.optimizer_construct_outer_f(self.model.outer_params())      
                self._dual_optim = True 
        if(optim2 is None):
            return optim
        return [optim, optim2]
    
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
    
    def training_step_acc(self, x, y, loss, latent, model_out_dict, optimizer_idx):
        self.log("train_loss/total", loss.item() + self._saved_loss)

        if(self._is_inner_enabled(optimizer_idx)):
            self.train_acc_inner(self._loss_f.classify(self._inner_output), y)
            self.log("train_step_acc_inner", self.train_acc_inner, on_step=False, on_epoch=True) 
            return loss
        
        if(self._is_outer_enabled(optimizer_idx)):
            self.train_acc(self._outer_loss_f.classify(latent), y)
            self.log("train_step_acc_outer", self.train_acc, on_step=False, on_epoch=True) 
            return loss
        raise Exception('Unacceptable state.')

    def get_obj_str_type(self) -> str:
        return 'ClLatentDual_' + super().get_obj_str_type()
        
class ClLatentDualHalved(ClLatentDual):

    @dataclass
    class Outer():
        @dataclass
        class HalfOptimizer():
            @dataclass
            class First():        
                type: str = None
                reset_type: str = None
                kwargs: dict = None

            @dataclass
            class Second():        
                type: str = None
                reset_type: str = None
                kwargs: dict = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.automatic_optimization = False

        ClLatentDual.__after_init__(self.cfg_outer_optim_first_half, self.cfg_optim)
        ClLatentDual.__after_init__(self.cfg_outer_optim_second_half, self.cfg_optim)

        outer_first_half_optim_manager = ModelOptimizerManager(
            optimizer_type=self.cfg_outer_optim_first_half.type,
        )
        outer_second_half_optim_manager = ModelOptimizerManager(
            optimizer_type=self.cfg_outer_optim_second_half.type,
        )

        self.optimizer_construct_first_half_f = outer_first_half_optim_manager.get_optimizer(**self.cfg_outer_optim_first_half.kwargs)
        self.optimizer_construct_second_half_f = outer_second_half_optim_manager.get_optimizer(**self.cfg_outer_optim_second_half.kwargs)

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg_outer_optim_first_half': ClLatentDualHalved.Outer.HalfOptimizer.First,
            'cfg_outer_optim_second_half': ClLatentDualHalved.Outer.HalfOptimizer.Second,
        })
        b.update({
            'outer.optim_half.first': 'cfg_outer_optim_first_half',
            'outer.optim_half.second': 'cfg_outer_optim_second_half',
        })
        return a, b

    def _create_optimizer(self) -> torch.optim.Optimizer:
        optims = super()._create_optimizer()
        halves = self._create_optimizer_halves()
        if(not self._dual_optim):
            raise Exception("Cannot have halved optimizers when only using one optimizer.")
        halves.append(optims[-1])
        return halves
            

    def _create_optimizer_halves(self):
        if(self.optimizer_construct_f is None):
            optim_first_half = torch.optim.Adam(self.model.model.first_half_params(), lr=optim_Adam_config["lr"])
            optim_second_half = torch.optim.Adam(self.model.model.second_half_params(), lr=optim_Adam_config["lr"])
        else:  
            optim_first_half = self.optimizer_construct_first_half_f(self.model.model.first_half_params())
            optim_second_half = self.optimizer_construct_second_half_f(self.model.model.second_half_params())
        return [optim_first_half, optim_second_half]

    def process_losses_normal(self, x, y, latent, log_label, optimizer_idx, model_out_dict=None):
        first_half_optim, second_half_optim, outer_optim = self.optimizers()
        if(self._is_inner_enabled(optimizer_idx)):
            loss_inner = self._loss_f(self._inner_output, y) * self.cfg_loss_chi_dual.inner_scale
            self.log(f"{log_label}/island_CHI-K", loss_inner)

            first_half_optim.zero_grad()
            second_half_optim.zero_grad()

            self.manual_backward(loss_inner)

            second_half_optim.step()
            first_half_optim.step()

            return loss_inner
        
        if(self._is_outer_enabled(optimizer_idx)):
            loss_outer = self._outer_loss_f(latent, y) * self.cfg_loss_chi_dual.outer_scale
            self.log(f"{log_label}/island_CROSS-E", loss_outer)

            outer_optim.zero_grad()
            self.manual_backward(loss_outer)
            outer_optim.step()

            return loss_outer
        raise Exception("Both losses (inner, outer) cannot be zero!")

    def training_step(self, batch, batch_idx):
        super().training_step(batch=batch, batch_idx=batch_idx, optimizer_idx=None)
        # do not return anything

