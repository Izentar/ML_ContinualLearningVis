import torch
from typing import Any, Dict
from model.overlay import cl_base
from torch import nn
from torch.nn.functional import cross_entropy
from robustness import model_utils
from loss_function.chiLoss import ChiLossBase, DummyLoss, BaseLoss
from utils import utils
from utils import pretty_print as pp

from config.default import datasets, datasets_map
from dataclasses import dataclass
from numbers import Number

class ClModel(cl_base.ClBase):
    @dataclass
    class Config(cl_base.ClBase.Config):
        train_sanity_check: bool = False

    @dataclass
    class Robust(utils.BaseConfigDataclass):
        dataset_name: str = None
        data_path: str = None
        resume_path: str = None
        enable: bool = False

        @dataclass
        class Kwargs(utils.BaseConfigDataclass):
            constraint: int = 2
            eps: float = 0.5
            step_size: float = 1.5
            iterations: int = 10
            random_start: int = 0
            random_restart: int = 0
            custom_loss: float = None
            use_worst: bool = True
            with_latent: bool = False
            fake_relu: bool = False
            no_relu: bool = False
            make_adversary: bool = False

    @dataclass
    class LayerReplace(utils.BaseConfigDataclass):
        enable: bool = False
        source: object = None
        destination_f: 'function' = None

    def __init__(
        self,
        model:nn.Module=None,
        loss_f:nn.Module=None,
        *args, 
        **kwargs
    ):    
        super().__init__(*args, **kwargs)

        self._robust_model_set = False
        self._setup_model(model=model)

        if(self.cfg_layer_replace.enable):
            if(self.cfg_layer_replace.source is None or self.cfg_layer_replace.destination_f is None):
                raise Exception(f'replace_layer_from is None: {self.cfg_layer_replace.source is None} or cfg_layer_replace.destination_f is None: {self.cfg_layer_replace.destination_f is None}')
            pp.sprint(f'{pp.COLOR.WARNING}INFO: Replacing layer from "{self.cfg_layer_replace.source.__class__.__name__}"')
            utils.replace_layer(self, 'model', self.cfg_layer_replace.source, self.cfg_layer_replace.destination_f)

        if(loss_f is not None):
            self._setup_loss_f(loss_f)

        self.save_hyperparameters(ignore=['model', '_loss_f', 'loss_f', 'optim_manager', 'cfg_map'])

    def _setup_loss_f(self, loss_f):
        self._loss_f = DummyLoss(loss_f) if not isinstance(loss_f, BaseLoss) else loss_f
        pp.sprint(f"{pp.COLOR.NORMAL_3}INFO: Using loss {str(self._loss_f)}")

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg_robust': ClModel.Robust,
            'cfg_robust_kwargs': ClModel.Robust.Kwargs,
            'cfg_layer_replace': ClModel.LayerReplace,
            'cfg': ClModel.Config,
        })

        b.update({
            'robust': 'cfg_robust',
            'robust.kwargs': 'cfg_robust_kwargs',
            'layer_replace': 'cfg_layer_replace',
        })
        return a, b

    def _setup_model(self, model):
        if(self.cfg_robust.enable):
            if(self.cfg_robust.dataset_name is not None and self.cfg_robust.data_path is not None):
                robust_dataset = self._get_dataset_list(self.cfg_robust.dataset_name)[1](data_path=self.cfg_robust.data_path)
                if(robust_dataset is not None and self.cfg_robust.enable):
                    pp.sprint(f'{pp.COLOR.WARNING}INFO: Enabled robust model overlay')
                    self.model = model_utils.make_and_restore_model(
                        arch=model, dataset=robust_dataset, resume_path=self.cfg_robust.resume_path
                    )[0]
                    self._robust_model_set = True
                    return
            raise Exception('Robust selected but robust_dataset not provided.')
        else:
            self.model = model

    @property
    def loss_f(self):
        return self._loss_f
    
    @property
    def name(self):
        return type(self).__name__

    def _get_dataset_list(name:str):
        if name is not None:
            cap = name.capitalize()
            if(cap in datasets_map):
                return datasets_map[cap]
        raise Exception(f"Unknown dataset {name}.")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def call_loss(self, input, target, train, **kwargs):
        return self._loss_f(input, target, train)
    
    def _weight_train_sanity_check(self):
        if(self.cfg.train_sanity_check):
            if(not hasattr(self, "_saved_tensor_sanity_check")):
                self._saved_tensor_sanity_check = self.model.get_objective_layer().weight.detach().cpu()
                self._sanity_check_counter = 0
            else:
                self._sanity_check_counter += 1
            if(self._sanity_check_counter == 1):
                test_sanity_val = self.model.get_objective_layer().weight.detach().cpu()
                if not torch.allclose(self._saved_tensor_sanity_check, test_sanity_val):
                    pp.sprint(f"{pp.COLOR.WARNING}DEBUG: SANITY CHECK: Model training success")
                    return
                pp.sprint(f'{pp.COLOR.WARNING}DEBUG: SANITY CHECK: FAIL: Model training fail sanity check. No weights update')
                pp.sprint(f"{pp.COLOR.WARNING}", self._saved_tensor_sanity_check)
                pp.sprint(f"{pp.COLOR.WARNING}", test_sanity_val)

    def training_step_normal_setup(self, x, y):
        if(self.cfg_robust.enable):
            model_out = self(
                x, target=y, **vars(self.cfg_robust_kwargs)
            )
        else:
            model_out = self(x)
        latent, model_out_dict = self.get_model_out_data(model_out)
        
        for k, v in self._loss_f.to_log.items():
            if(isinstance(v, Number)):
                self.log(k, float(v))
            else:
                self.log(k, v)

        return latent, model_out_dict

    def training_step_normal(self, batch, optimizer_idx):
        x, y = batch
        latent, model_out_dict = self.training_step_normal_setup(x=x, y=y)

        self._weight_train_sanity_check()

        loss = self.process_losses_normal(
            x=x, 
            y=y, 
            latent=latent, 
            model_out_dict=model_out_dict,
            log_label="train_loss", 
            optimizer_idx=optimizer_idx,
        )
        self.training_step_acc_normal(
            x=x, y=y,
            loss=loss, latent=latent, 
            model_out_dict=model_out_dict, optimizer_idx=optimizer_idx
        )
        return loss
    
    def classify(self, predictions):
        return self._loss_f.classify(predictions)

    def training_step_acc_normal(self, x, y, loss, latent, model_out_dict, optimizer_idx):
        self.log("train_loss/total", loss)
        self.train_acc(self._loss_f.classify(latent), y)
        self.log("train_step_acc", self.train_acc, on_step=False, on_epoch=True)

    def process_losses_normal(self, x, y, latent, log_label, optimizer_idx, model_out_dict=None):
        loss = self._loss_f(latent, y)
        self.log(f"{log_label}/classification_loss", loss)
        return loss
    
    def training_step_dream_setup(self, x, y):
        if(self.cfg_robust.enable):
            model_out = self(
                x, target=y, **vars(self.cfg_robust_kwargs)
            )
        else:
            model_out = self(x)
        latent, model_out_dict = self.get_model_out_data(model_out)

        return latent, model_out_dict
    
    def training_step_dream(self, batch, optimizer_idx):
        x, y = batch
        latent, model_out_dict = self.training_step_dream_setup(x=x, y=y)

        loss = self.process_losses_dreams(
            x=x, 
            y=y, 
            latent=latent, 
            model_out_dict=model_out_dict, 
            log_label="train_loss_dream",
            optimizer_idx=optimizer_idx,
        )
        self.training_step_acc_dream(
            x=x, y=y,
            loss=loss, latent=latent, 
            model_out_dict=model_out_dict, optimizer_idx=optimizer_idx
        )
        
        return loss
    
    def training_step_acc_dream(self, x, y, loss, latent, model_out_dict, optimizer_idx):
        self.log("train_loss_dream/total", loss)
        self.train_acc_dream(self._loss_f.classify(latent), y)
        self.log("train_step_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)

    def process_losses_dreams(self, x, y, latent, log_label, optimizer_idx, model_out_dict=None):
        return self.process_losses_normal(
            x=x, 
            y=y, 
            latent=latent, 
            model_out_dict=model_out_dict,
            log_label=log_label,
            optimizer_idx=optimizer_idx,
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        val_loss = cross_entropy(self._loss_f.classify(latent), y)
        self.log("val_last_step_loss", val_loss, on_epoch=True)
        valid_acc = self.valid_accs(dataloader_idx)
        valid_acc(latent, y)
        self.log("valid_step_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        test_loss = cross_entropy(self._loss_f.classify(latent), y)
        self.log("test_loss", test_loss, on_step=True)
        self.test_acc(self._loss_f.classify(latent), y)
        self.log("test_step_acc", self.test_acc)

    def get_objective_target_name(self) -> str:
        if(self.cfg_robust.enable):
            return "model.model." + self.model.model.get_objective_layer_name()
        return "model." + self.model.get_objective_layer_name()

    def get_objective_layer(self):
        if(self.cfg_robust.enable):
            return self.model.model.get_objective_layer()
        return self.model.get_objective_layer()

    def get_objective_layer_output_shape(self):
        if(self.cfg_robust.enable):
            return self.model.model.get_objective_layer_output_shape()
        return self.model.get_objective_layer_output_shape()

    def get_root_objective_target(self): 
        if(self.cfg_robust.enable):
            return "model.model." + self.model.model.get_root_name()
        return "model." + self.model.get_root_name()

    def loss_to(self, device):
        """
            Cast loss class (all of their tensors) to given device.
        """
        self._loss_f.to(device)

    def get_obj_str_type(self) -> str:
        if(self.cfg_robust.enable):
            return 'CLModel_' + type(self.model.model).__qualname__
        else:
            return 'CLModel_' + type(self.model).__qualname__

    def init_weights(self):
        if(not self._robust_model_set):
            self.model._initialize_weights()
