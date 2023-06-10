from typing import Any, Dict
from model import base
import torch
from torch import nn, sigmoid
from torch.nn.functional import relu, cross_entropy, mse_loss
from config.default import optim_Adam_config
from torch.autograd.variable import Variable
from robustness import model_utils
from loss_function.chiLoss import ChiLoss, l2_latent_norm, OneHot
from utils.data_manipulation import select_class_indices_tensor
from utils.cyclic_buffer import CyclicBufferByClass
from loss_function.chiLoss import ChiLossBase, DummyLoss, BaseLoss
from model.SAE import SAE_CIFAR
from utils import utils
import wandb
import pandas as pd
from utils import pretty_print as pp
import torchmetrics

from config.default import datasets, datasets_map
from dataclasses import dataclass, field
from utils.utils import search_kwargs

class CLModel(base.CLBase):
    @dataclass
    class Robust():
        dataset_name: str = None
        data_path: str = None
        resume_path: str = None
        enable: bool = False

        @dataclass
        class Kwargs():
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
    class LayerReplace():
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

        self.save_hyperparameters(ignore=['model', '_loss_f', 'loss_f', 'optim_manager', 
        'optimizer_construct_f', 'scheduler_construct_f', 'optimizer_restart_params_f', 'cfg_map'])

    def _setup_loss_f(self, loss_f):
        self._loss_f = DummyLoss(loss_f) if not isinstance(loss_f, BaseLoss) else loss_f
        pp.sprint(f"{pp.COLOR.NORMAL_3}INFO: Using loss {str(self._loss_f)}")

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg_robust': CLModel.Robust,
            'cfg_robust_kwargs': CLModel.Robust.Kwargs,
            'cfg_layer_replace': CLModel.LayerReplace,
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

    def training_step_normal(self, batch, optimizer_idx):
        x, y = batch

        if(self.cfg_robust.enable):
            model_out = self(
                x, target=y, **vars(self.cfg_robust_kwargs)
            )
        else:
            model_out = self(x)
        latent, model_out_dict = self.get_model_out_data(model_out)
        
        #a = torch.abs(latent.detach()).sum().cpu().item()
        #self.log("train_loss/latent_model_abs_sum", a)
        for k, v in self._loss_f.to_log.items():
            self.log(k, v)

        loss = self.process_losses_normal(
            x=x, 
            y=y, 
            latent=latent, 
            model_out_dict=model_out_dict,
            log_label="train_loss", 
        )
        self.training_step_acc(
            x=x, y=y,
            loss=loss, latent=latent, 
            model_out_dict=model_out_dict, optimizer_idx=optimizer_idx
        )
        return loss

    def training_step_acc(self, x, y, loss, latent, model_out_dict, optimizer_idx):
        self.log("train_loss/total", loss)
        self.train_acc(self._loss_f.classify(latent), y)
        self.log("train_step_acc", self.train_acc, on_step=False, on_epoch=True)

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        loss = self._loss_f(latent, y)
        self.log(f"{log_label}/classification_loss", loss)
        return loss

    def training_step_dream(self, batch):
        x, y = batch

        if(self.cfg_robust.enable):
            model_out = self(
                x, target=y, **vars(self.cfg_robust_kwargs)
            )
        else:
            model_out = self(x)
        latent, model_out_dict = self.get_model_out_data(model_out)

        loss = self.process_losses_dreams(
            x=x, 
            y=y, 
            latent=latent, 
            model_out_dict=model_out_dict, 
            log_label="train_loss_dream",
        )
        self.log("train_loss_dream/total", loss)
        self.train_acc_dream(self._loss_f.classify(latent), y)
        self.log("train_step_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)
        return loss

    def process_losses_dreams(self, x, y, latent, log_label, model_out_dict=None):
        return self.process_losses_normal(
            x=x, 
            y=y, 
            latent=latent, 
            model_out_dict=model_out_dict,
            log_label=log_label 
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        val_loss = cross_entropy(self._loss_f.classify(latent), y)
        self.log("val_last_step_loss", val_loss, on_epoch=True)
        valid_acc = self.valid_accs(dataloader_idx)
        valid_acc(latent, y)
        self.log("valid_step_acc", valid_acc.compute())

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
        self._loss_f.to(device)

    def get_obj_str_type(self) -> str:
        if(self.cfg_robust.enable):
            return 'CLModel_' + type(self.model.model).__qualname__
        else:
            return 'CLModel_' + type(self.model).__qualname__

    def init_weights(self):
        if(not self._robust_model_set):
            self.model._initialize_weights()

class CLLatent(CLModel):
    @dataclass
    class Latent():
        size: int = None

        def post_init_Latent(self, num_classes):
            self.size = self.size if self.size is not None else num_classes

        @dataclass
        class Buffer():
            size_per_class: int = 40

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg_latent.post_init_Latent(self.cfg.num_classes)
        

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg_latent': CLLatent.Latent,
            'cfg_latent_buffer': CLLatent.Latent.Buffer,
        })

        b.update({
            'latent': 'cfg_latent',
            'latent.buffer': 'cfg_latent_buffer',
        })
        return a, b

class CLModelIslandsOneHot(CLLatent):
    @dataclass
    class Latent(CLLatent.Latent):
        @dataclass
        class OneHot():
            type: str = None
            scale: float = 1.
            special_class: int = 1

    def __init__(self, *args, **kwargs):
        kwargs.pop('loss_f', None)
        super().__init__(*args, loss_f=torch.nn.MSELoss(), **kwargs)
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=self.cfg.num_classes, dimensions=self.cfg_latent.size, size_per_class=self.cfg_latent_buffer.size_per_class)
        
        onehot_means = self.get_onehots(self.cfg_onehot.type)
        onehot_means_to_print = []
        for k, v in onehot_means.items():
            onehot_means_to_print.insert(k, v)
        pp.sprint(f"{pp.COLOR.NORMAL}INFO: Selected onehot type: {self.cfg_onehot.type}\nmeans:\n{torch.stack(onehot_means_to_print)}\n")
        self._loss_f = OneHot(onehot_means, self.cyclic_latent_buffer, loss_f=self._loss_f)

        self.valid_correct = 0
        self.valid_all = 0

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg_onehot': CLModelIslandsOneHot.Latent.OneHot,
            #'': CLModelIslandsOneHot.Config,
            'cfg_latent': CLModelIslandsOneHot.Latent,
        })

        b.update({
            'latent.onehot': 'cfg_onehot',
            'latent': 'cfg_latent',
        })
        return a, b

    def get_onehots(self, mytype):
        if(mytype == 'one_cl'):
            d = {}
            for i in range(self.cfg.num_classes):
                d[i] = torch.zeros((self.cfg_latent.size,), dtype=torch.float)
            for k, v in d.items():
                if(k == self.cfg_onehot.special_class):
                    v[self.cfg_latent.size-1] = 1 * self.cfg_onehot.scale
                else:
                    v[0] = 1 * self.cfg_onehot.scale
            return d
        elif(mytype == 'diagonal'):
            d = {}
            counter = 0
            step = 0
            for i in range(self.cfg.num_classes):
                d[i] = torch.zeros((self.cfg_latent.size,), dtype=torch.float)
                if(counter % self.cfg_latent.size == 0):
                    step += 1
                for s in range(step):
                    d[i][(counter + s) % self.cfg_latent.size] = 1 * self.cfg_onehot.scale
                counter += 1
            return d
        elif(mytype == 'random'):
            classes = {}
            for s in range(self.cfg.num_classes):
                classes[s] = torch.rand(self.cfg_latent.size) * self.cfg_onehot.scale
            return classes
        else:
            raise Exception(f"Bad value, could not find means in OneHot config. Key: {mytype}")

    def call_loss(self, input, target, train, **kwargs):
        return self._loss_f(input, target, train)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def decode(self, target):
        return self._loss_f.decode(target)

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        loss = self._loss_f(latent, y)
        self.log(f"{log_label}/MSE_loss", loss)

        # log values of a point
        selected_class = 0
        uniq = torch.unique(y)
        cl_batch = None
        for u in uniq:
            if(u.item() == selected_class):
                cl_batch = latent[y == u]
                break
        if(cl_batch is not None):
            cl_batch = torch.mean(cl_batch, dim=0)
            for idx, m in enumerate(cl_batch):
                self.log(f"val_cl0_dim{idx}", m)

        return loss

    def process_losses_dreams(self, x, y, latent, log_label, model_out_dict=None):
        return self.process_losses_normal(
            x=x, 
            y=y, 
            latent=latent, 
            model_out_dict=model_out_dict,
            log_label=log_label 
        )

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        model_out = self(x)
        y_model, _ = self.get_model_out_data(model_out)
        val_loss = self._loss_f(y_model, y)
        self.log("val_last_step_loss", val_loss, on_epoch=True)

        classified_to_class = self._loss_f.classify(y_model)
        valid_acc = self.valid_accs(dataloader_idx)
        valid_acc(classified_to_class, y)
        self.log("valid_acc", valid_acc.compute())

    def valid_to_class(self, classified_to_class, y):
        uniq = torch.unique(classified_to_class)
        for i in uniq:
            selected = (y == i)
            correct_sum = torch.logical_and((classified_to_class == i), selected).sum().item()
            target_sum_total = selected.sum().item()
            if(target_sum_total != 0):
                pass
                pp.sprint(f'{pp.COLOR.NORMAL_3}Cl {i.item()}: {correct_sum / target_sum_total}')
            self.valid_correct += correct_sum
            self.valid_all += target_sum_total
        pp.sprint(f"{pp.COLOR.NORMAL_3}", self.valid_correct, self.valid_all, self.valid_correct/self.valid_all)

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        y_model, _ = self.get_model_out_data(model_out)
        test_loss = self._loss_f(y_model, y)
        self.log("test_loss", test_loss, on_step=True)

        classified_to_class = self._loss_f.classify(y_model)
        self.test_acc(classified_to_class, y)
        #self.valid_to_class(classified_to_class, y)
        self.log("test_step_acc", self.test_acc)

    def get_obj_str_type(self) -> str:
        return 'CLModelIslandsTest_' + super().get_obj_str_type()

class CLModelWithIslands(CLLatent):
    @dataclass
    class Config(CLModel.Config):
        norm_lambda: float = 0.

    @dataclass
    class Loss():
        @dataclass
        class Chi():
            sigma: float = 0.2
            rho: float = 0.2

    def __init__(
            self, 
            *args, 
            buff_on_same_device=False,  
            **kwargs
        ):
        kwargs.pop('loss_f', None)
        super().__init__(*args, **kwargs)
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=self.cfg.num_classes, dimensions=self.cfg_latent.size, size_per_class=self.cfg_latent_buffer.size_per_class)
        loss_f = ChiLoss(sigma=self.cfg_loss_chi.sigma, rho=self.cfg_loss_chi.rho, cyclic_latent_buffer=self.cyclic_latent_buffer, loss_means_from_buff=False)
        self._setup_loss_f(loss_f)


        self.norm = l2_latent_norm
        self.buff_on_same_device = buff_on_same_device

        self._means_once = False

    def _get_config_maps(self):
        a, b = super()._get_config_maps()
        a.update({
            'cfg': CLModelWithIslands.Config,
            'cfg_latent': CLModelWithIslands.Latent,
            'cfg_loss_chi': CLModelWithIslands.Loss.Chi,
        })
        b.update({
            'latent': 'cfg_latent',
            'loss.chi': 'cfg_loss_chi',
        })
        return a, b

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        loss = self._loss_f(latent, y)

        if(self.cfg.norm_lambda != 0.):
            norm = self.norm(latent, self.cfg.norm_lambda)
            self.log(f"{log_label}/norm", norm)
            loss += norm
        self.log(f"{log_label}/island", loss)

        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        val_loss = self._loss_f(latent, y, train=False)
        self.log("val_last_step_loss", val_loss, on_epoch=True)
        valid_acc = self.valid_accs(dataloader_idx)
        valid_acc(self._loss_f.classify(latent), y)
        self.log("valid_acc", valid_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        test_loss = self._loss_f(latent, y, train=False)
        self.log("test_loss", test_loss, on_step=True)

        classified_to_class = self._loss_f.classify(latent)
        self.test_acc(classified_to_class, y)
        self.log("test_acc", self.test_acc)

        self._test_step_log_data()

    def _test_step_log_data(self):
        # log additional data only once
        if(not self._means_once):
            new_means = {str(k): v for k, v in self.loss_f.cloud_data.mean().items()}
            distance_key, dist_val = self._calc_distance_to_each_other(new_means)
            table_distance = wandb.Table(columns=distance_key, data=[dist_val])
            wandb.log({'chiloss/distance_to_means': table_distance})

            means_val = self._parse_bar_plot(new_means)
            for k, v in means_val.items():
                table_means = wandb.Table(data=v, columns=['dimension', 'value'])
                wandb.log({f'chiloss/means_cl-{k}': wandb.plot.bar(table_means, label='dimension', value='value', title=f"Class-{k}")})

            new_std = {str(k): v for k, v in self.loss_f.cloud_data.std().items()}
            std_key_class, std_val = self._parse_std(new_std)
            table_std = wandb.Table(columns=std_key_class, data=[std_val])
            wandb.log({'chiloss/std_for_classes': table_std})
            
            self._means_once = True

    def load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().load_checkpoint(checkpoint)
        if(loss_f := checkpoint.get('loss_f')):
            self._loss_f = loss_f
        else:
            pp.sprint(f"{pp.COLOR.WARNING}WARNING: model loss function not loaded. Using default constructor.")
        if(buffer := checkpoint.get('buffer')):
            self.cyclic_latent_buffer = buffer
        else:
            pp.sprint(f"{pp.COLOR.WARNING}WARNING: model latent buffer not loaded. Using default constructor.")
    
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        super().on_save_checkpoint(checkpoint)
        checkpoint['loss_f'] = self._loss_f
        checkpoint['buffer'] = self.cyclic_latent_buffer

    def _parse_std(self, data:dict):
        key = []
        value = []
        for k, v in data.items():
            for k2, v2 in enumerate(v):
                key.append(f"cl={k}::dim={k2}")
                value.append(v2)
        return key, value

    def _parse_bar_plot(self, data:dict):
        vals = {}
        for k, v in data.items():
                vals[k] = []
                for k2, v2 in enumerate(v):
                    vals[k].append([k2, v2])
        return vals

    def _calc_distance_to_each_other(self, means):
        pdist = torch.nn.PairwiseDistance(p=2)
        keys = []
        vals = []
        for k, v in means.items():
            for k2, v2 in means.items():
                keys.append(f"{k}:{k2}")
                vals.append(pdist(v, v2))
        return keys, vals

    def training_step(self, batch, batch_idx):
        if(self.cyclic_latent_buffer is not None and self.buff_on_same_device):
            self.cyclic_latent_buffer.to(self.device)
        return super().training_step(batch=batch, batch_idx=batch_idx)

    def get_obj_str_type(self) -> str:
        return 'CLModelWithIslands_' + super().get_obj_str_type()
        
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
        return "ln"

    def get_root_name(self):
        return ""

    def get_objective_layer(self):
        return self.ln

    def get_objective_layer_output_shape(self):
        return (self.ln.out_features,)
    
    def outer_params(self):
        return [self.ln.weight, self.ln2.weight, self.ln3.weight]
    
    @property
    def name(self):
        return self.model.name

class CLModelLatentDual(CLModelWithIslands):
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
            'cfg_loss_chi_dual': CLModelLatentDual.Loss.Chi.Dual,
        })
        b.update({
            'loss.chi.dual': 'cfg_loss_chi_dual',
        })
        return a, b

    def __init__(
            self, 
            *args,
            model:nn.Module=None,
            **kwargs
        ):
        model = ModelSufix(model)
        super().__init__(*args, model=model, **kwargs)
        self._outer_loss_f = DummyLoss(torch.nn.CrossEntropyLoss())

        self._hook_handle = model.get_objective_layer().register_forward_hook(self._hook)

        self.valid_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes).to(self.device)
        self.test_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        self.train_acc_inner = torchmetrics.Accuracy(task='multiclass', num_classes=self.cfg.num_classes)
        

    def _hook(self, module, input, output):
        self._first_output = output

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        if(self._optimizer_idx == 0):
            loss_inner = self._loss_f(self._first_output, y) * self.cfg_loss_chi_dual.inner_scale
            self.log(f"{log_label}/island_CHI-K", loss_inner)
            return loss_inner
        
        loss = self._outer_loss_f(latent, y) * self.cfg_loss_chi_dual.outer_scale
        self.log(f"{log_label}/island_CROSS-E", loss)
        return loss
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        if(self.optimizer_construct_f is None):
            optim = torch.optim.Adam(self.model.model.parameters(), lr=optim_Adam_config["lr"])
            optim2 = torch.optim.Adam(self.model.outer_params(), lr=optim_Adam_config["lr"])
        else:
            optim = self.optimizer_construct_f(self.model.model.parameters())        
            optim2 = self.optimizer_construct_f(self.model.outer_params())        
        self.optimizer = optim
        return [optim, optim2]

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        val_loss_outer = self._outer_loss_f(latent, y, train=False)
        val_loss_inner = self._loss_f(latent, y, train=False)
        self.log("val_last_step_loss_outer", val_loss_outer, on_epoch=True)
        self.log("val_last_step_loss_inner", val_loss_inner, on_epoch=True)

        valid_acc = self.valid_accs(dataloader_idx)
        valid_acc(self._outer_loss_f.classify(latent), y)
        self.log("valid_acc_outer", valid_acc.compute())

        self.valid_acc_inner(self._loss_f.classify(self._first_output), y)
        self.log("valid_acc_inner", self.valid_acc_inner.compute())

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        test_loss_inner = self._loss_f(self._first_output, y, train=False)
        self.log("test_loss_inner", test_loss_inner)

        test_loss_outer = self._outer_loss_f(latent, y, train=False)
        self.log("test_loss_outer", test_loss_outer)

        self.test_acc_inner(self._loss_f.classify(self._first_output), y)
        self.log("test_acc_inner", self.test_acc)

        self.test_acc(self._outer_loss_f.classify(latent), y)
        self.log("test_acc_outer", self.test_acc)

        self._test_step_log_data()

    def training_step(self, batch, batch_idx, optimizer_idx):
        self._optimizer_idx = optimizer_idx
        loss = super().training_step(batch=batch, batch_idx=batch_idx)
        if(optimizer_idx == 0):
            self._saved_loss = loss.item()
        return loss
    
    def training_step_acc(self, x, y, loss, latent, model_out_dict, optimizer_idx):
        if(optimizer_idx == 0):
            self.train_acc_inner(self._loss_f.classify(latent), y)
            self.log("train_step_acc_inner", self.train_acc_inner, on_step=False, on_epoch=True) 
        if(optimizer_idx == 1):
            self.log("train_loss/total", loss.item() + self._saved_loss)
            self.train_acc(self._outer_loss_f.classify(latent), y)
            self.log("train_step_acc_outer", self.train_acc, on_step=False, on_epoch=True) 
        
        return loss