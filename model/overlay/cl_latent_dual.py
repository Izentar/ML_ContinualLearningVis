from typing import Any, Dict
import torch
from config.default import optim_Adam_config
from loss_function.chiLoss import DummyLoss
from utils import pretty_print as pp
from torch.nn.functional import relu
import torchmetrics

from dataclasses import dataclass
from model.overlay.cl_latent_chi import ClLatentChi

class ModelSufix(torch.nn.Module):
    def __init__(self, model) -> None:
        super().__init__()
        self.model = model

        self.ln = torch.nn.Linear(self.model.get_objective_layer().out_features, self.model.get_objective_layer().out_features)
        self.ln2 = torch.nn.Linear(self.model.get_objective_layer().out_features, self.model.get_objective_layer().out_features)
        self.ln3 = torch.nn.Linear(self.model.get_objective_layer().out_features, self.model.get_objective_layer().out_features)
    
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

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg_loss_chi_dual': ClLatentDual.Loss.Chi.Dual,
        })
        b.update({
            'loss.chi.dual': 'cfg_loss_chi_dual',
        })
        return a, b

    def __init__(
            self, 
            *args,
            model:torch.nn.Module=None,
            **kwargs
        ):
        model = ModelSufix(model)
        super().__init__(*args, model=model, **kwargs)
        self._outer_loss_f = DummyLoss(torch.nn.CrossEntropyLoss())

        self._hook_handle = model.model.get_objective_layer().register_forward_hook(self._hook)

        self.valid_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes).to(self.device)
        self.test_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self.train_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self._saved_loss = 0.0
        
        if(self.cfg_loss_chi_dual.inner_scale == 0.):
            pp.sprint(f"{pp.COLOR.NORMAL}INFO: {self.name} inner scale set to zero.")
        if(self.cfg_loss_chi_dual.outer_scale == 0.):
            pp.sprint(f"{pp.COLOR.NORMAL}INFO: {self.name} outer scale set to zero.")
        if(self.cfg_loss_chi_dual.inner_scale == 0. and self.cfg_loss_chi_dual.outer_scale == 0.):
            raise Exception("Both losses (inner, outer) cannot be zero!")

    def _hook(self, module, input, output):
        self._inner_output = output

    def _is_inner_enabled(self):
        return self._optimizer_idx == 0 and self.cfg_loss_chi_dual.inner_scale != 0. or (
            self.cfg_loss_chi_dual.outer_scale == 0.
        )

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        if(self._is_inner_enabled()):
            loss_inner = self._loss_f(self._inner_output, y) * self.cfg_loss_chi_dual.inner_scale
            self.log(f"{log_label}/island_CHI-K", loss_inner)
            return loss_inner
        
        if(self.cfg_loss_chi_dual.outer_scale != 0):
            loss_outer = self._outer_loss_f(latent, y) * self.cfg_loss_chi_dual.outer_scale
            self.log(f"{log_label}/island_CROSS-E", loss_outer)
            return loss_outer
        raise Exception("Both losses (inner, outer) cannot be zero!")
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        optim2 = None
        if(self.optimizer_construct_f is None):
            if(self.cfg_loss_chi_dual.inner_scale == 0. or self.cfg_loss_chi_dual.outer_scale == 0.):
                optim = torch.optim.Adam(self.model.parameters(), lr=optim_Adam_config["lr"])
            else:
                optim = torch.optim.Adam(self.model.model.parameters(), lr=optim_Adam_config["lr"])
                optim2 = torch.optim.Adam(self.model.outer_params(), lr=optim_Adam_config["lr"])
        else:
            if(self.cfg_loss_chi_dual.inner_scale == 0. or self.cfg_loss_chi_dual.outer_scale == 0.):
                optim = self.optimizer_construct_f(self.model.parameters()) 
            else:
                optim = self.optimizer_construct_f(self.model.model.parameters())        
                optim2 = self.optimizer_construct_f(self.model.outer_params())        
        self.optimizer = optim
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
        if(self.cfg_loss_chi_dual.outer_scale == 0.):
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
        self._optimizer_idx = 0
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
        if(self.cfg_loss_chi_dual.outer_scale == 0.):
            return
        test_loss_outer = self._outer_loss_f(latent, y, train=False)
        self.log("test_loss_outer", test_loss_outer)

        self.test_acc(self._outer_loss_f.classify(latent), y)
        self.log("test_acc_outer", self.test_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        self._optimizer_idx = 0
        self._test_step_inner(y)
        self._test_step_outer(latent=latent, y=y)

        self._test_step_log_data()

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        self._optimizer_idx = optimizer_idx
        loss = super().training_step(batch=batch, batch_idx=batch_idx, optimizer_idx=optimizer_idx)
        self._saved_loss = 0.0
        if(self.cfg_loss_chi_dual.inner_scale != 0. and self.cfg_loss_chi_dual.outer_scale != 0.):
            self._saved_loss = loss.item()
        return loss
    
    def training_step_acc(self, x, y, loss, latent, model_out_dict, optimizer_idx):
        self.log("train_loss/total", loss.item() + self._saved_loss)

        if(self._is_inner_enabled()):
            self.train_acc_inner(self._loss_f.classify(self._inner_output), y)
            self.log("train_step_acc_inner", self.train_acc_inner, on_step=False, on_epoch=True) 
            return loss
        
        if(self.cfg_loss_chi_dual.outer_scale != 0.):
            self.train_acc(self._outer_loss_f.classify(latent), y)
            self.log("train_step_acc_outer", self.train_acc, on_step=False, on_epoch=True) 
            return loss
        raise Exception('Unacceptable state.')
