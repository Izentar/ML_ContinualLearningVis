from model import base
import torch
from torch import nn, sigmoid
from torch.nn.functional import relu, cross_entropy, mse_loss
from torch.autograd.variable import Variable
from robustness import model_utils
from loss_function.chiLoss import ChiLoss, l2_latent_norm, OneHot
from utils.data_manipulation import select_class_indices_tensor
from utils.cyclic_buffer import CyclicBufferByClass
from loss_function.chiLoss import ChiLossBase, DummyLoss
from model.SAE import SAE_CIFAR
from utils import utils
import wandb
import pandas as pd

from config.default import datasets, datasets_map

class SAE_standalone(base.CLBase):
    def __init__(self, num_classes, loss_f=None, reconstruction_loss_f=None, enable_robust=False, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.model = SAE_CIFAR(num_classes)
        self.enable_robust = enable_robust

        self._loss_f = loss_f if loss_f is not None else cross_entropy
        self.reconstruction_loss_f = reconstruction_loss_f if reconstruction_loss_f is not None else mse_loss

    def training_step_normal(self, batch):
        x, y = batch
        y_hat, y_reconstruction = self(x)
        loss_classification = self._loss_f(y_hat, y)
        loss_reconstruction = self.reconstruction_loss_f(y_reconstruction, Variable(x))
        alpha = 0.05
        loss = alpha * loss_classification + (1 - alpha) * loss_reconstruction
        self.log("train_loss/total", loss)
        self.log("train_loss/classification", loss_classification)
        self.log("train_loss/reconstrucion", loss_reconstruction)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def training_step_dream(self, batch):
        x, y = batch
        y_hat, y_reconstruction = self(x)
        loss_classification = self._loss_f(y_hat, y)
        loss_reconstruction = self.reconstruction_loss_f(y_reconstruction, Variable(x))
        alpha = 0.05
        loss = alpha * loss_classification + (1 - alpha) * loss_reconstruction
        self.log("train_loss_dream/total", loss)
        self.log("train_loss_dream/classification", loss_classification)
        self.log("train_loss_dream/reconstrucion", loss_reconstruction)
        self.train_acc_dream(y_hat, y)
        self.log("train_step_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat, _ = self(x)
        val_loss = self._loss_f(y_hat, y)
        self.log("val_last_step_loss", val_loss, on_epoch=True)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(y_hat, y)
        self.log("valid_step_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        test_loss = self._loss_f(y_hat, y)
        self.log("test_loss", test_loss, on_step=True)
        self.test_acc(y_hat, y)
        self.log("test_step_acc", self.test_acc)

    def get_objective_target_name(self) -> str:
        return "model.fc"

    def get_root_objective_target(self): 
        if(self.enable_robust):
            return "model.model." + self.model.model.get_root_name()
        return "model." + self.model.get_root_name()

    def forward(self, *args):
        return self.model(*args)

    def name(self):
        return str(self.model.__class__.__name__)

class CLModel(base.CLBase):
    def __init__(
        self,
        model:nn.Module=None,
        loss_f:nn.Module=None,
        robust_dataset_name:str=None,
        robust_data_path:str=None,
        attack_kwargs:dict=None,
        resume_path:str=None,
        enable_robust:bool=False,
        replace_layer=None,
        replace_layer_from=None,
        replace_layer_to_f=None,
        *args, 
        **kwargs
    ):
    
        super().__init__(*args, **kwargs)

        self.attack_kwargs = attack_kwargs
        self.enable_robust = enable_robust
        self.robust_data_path = robust_data_path
        self.robust_dataset_name = robust_dataset_name
        self.resume_path = resume_path
        self._robust_model_set = False
        self._setup_model(model=model, enable_robust=enable_robust, robust_data_path=robust_data_path)

        if(replace_layer):
            if(replace_layer_from is None or replace_layer_to_f is None):
                raise Exception(f'replace_layer_from is None: {replace_layer_from is None} or replace_layer_to_f is None: {replace_layer_to_f is None}')
            print(f'INFO: Replacing layer from "{replace_layer_from.__class__.__name__}d"')
            utils.replace_layer(self, 'model', replace_layer_from, replace_layer_to_f)
            
        self._loss_f = loss_f if isinstance(loss_f, ChiLossBase) else DummyLoss(loss_f)
        print(f"INFO: Using loss {str(self._loss_f)}")

        self.save_hyperparameters(ignore=['model', '_loss_f', 'optim_manager', 
        'optimizer_construct_f', 'scheduler_construct_f', 'optimizer_restart_params_f'])

    def _setup_model(self, model, enable_robust, robust_data_path):
        if(enable_robust):
            if(self.robust_dataset_name is not None and robust_data_path is not None):
                robust_dataset = self._get_dataset_list(self.robust_dataset_name)[1](data_path=robust_data_path)
                if(robust_dataset is not None and self.attack_kwargs is not None):
                    print('INFO: Enabled robust model overlay')
                    self.model = model_utils.make_and_restore_model(
                        arch=model, dataset=robust_dataset, resume_path=self.resume_path
                    )[0]
                    self._robust_model_set = True
                    return
            raise Exception('Robust selected but robust_dataset or attack_kwargs not provided.')
        else:
            self.model = model

    @property
    def loss_f(self):
        return self._loss_f

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

    def training_step_normal(self, batch):
        x, y = batch

        if(self.enable_robust):
            model_out = self(
                x, target=y, **self.attack_kwargs
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
        self.log("train_loss/total", loss)
        self.train_acc(self._loss_f.classify(latent), y)
        self.log("train_step_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        loss = self._loss_f(latent, y)
        self.log(f"{log_label}/classification_loss", loss)
        return loss

    def training_step_dream(self, batch):
        x, y = batch

        if(self.enable_robust):
            model_out = self(
                x, target=y, **self.attack_kwargs
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
        valid_acc = self.valid_accs[dataloader_idx]
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
        if(self.enable_robust):
            return "model.model." + self.model.model.get_objective_layer_name()
        return "model." + self.model.get_objective_layer_name()

    def get_objective_layer(self):
        if(self.enable_robust):
            return self.model.model.get_objective_layer()
        return self.model.get_objective_layer()

    def get_objective_layer_output_shape(self):
        if(self.enable_robust):
            return self.model.model.get_objective_layer_output_shape()
        return self.model.get_objective_layer_output_shape()

    def get_root_objective_target(self): 
        if(self.enable_robust):
            return "model.model." + self.model.model.get_root_name()
        return "model." + self.model.get_root_name()

    def loss_to(self, device):
        self._loss_f.to(device)

    def get_obj_str_type(self) -> str:
        return 'CLModel_' + type(self.model).__qualname__

    def name(self):
        return str(self.model.__class__.__name__)

    def init_weights(self):
        if(not self._robust_model_set):
            self.model._initialize_weights()

class CLModelIslandsTest(CLModel):
    def __init__(self, *args, hidden=10, num_classes=10, one_hot_means=None, size_per_class=40, **kwargs):
        kwargs.pop('loss_f', None)
        super().__init__(num_classes=num_classes, *args, loss_f=torch.nn.MSELoss(), **kwargs)
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=num_classes, dimensions=hidden, size_per_class=size_per_class)
        
        self.one_hot_means = one_hot_means
        self._loss_f = OneHot(one_hot_means, self.cyclic_latent_buffer, loss_f=self._loss_f)

        self.valid_correct = 0
        self.valid_all = 0

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
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(classified_to_class, y)
        self.log("valid_acc", valid_acc)

    def valid_to_class(self, classified_to_class, y):
        uniq = torch.unique(classified_to_class)
        for i in uniq:
            selected = (y == i)
            correct_sum = torch.logical_and((classified_to_class == i), selected).sum().item()
            target_sum_total = selected.sum().item()
            if(target_sum_total != 0):
                pass
                print(f'Cl {i.item()}: {correct_sum / target_sum_total}')
            self.valid_correct += correct_sum
            self.valid_all += target_sum_total
        print(self.valid_correct, self.valid_all, self.valid_correct/self.valid_all)

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
        if(self.enable_robust):
            return 'CLModelIslandsTest_' + type(self).__qualname__ + '_' + type(self.model.model).__qualname__
        return 'CLModelIslandsTest_' + type(self).__qualname__ + '_' + type(self.model).__qualname__

class CLModelWithIslands(CLModel):
    def __init__(
            self, 
            hidden, 
            loss_chi_buffer_size_per_class, 
            num_classes, 
            *args, 
            buff_on_same_device=False,  
            alpha=0.0, 
            norm_lambda=0.001, 
            sigma=0.2, 
            rho=1., 
            **kwargs
        ):
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=num_classes, dimensions=hidden, size_per_class=loss_chi_buffer_size_per_class)
        kwargs.pop('loss_f', None)
        super().__init__(
            loss_f=ChiLoss(sigma=sigma, rho=rho, cyclic_latent_buffer=self.cyclic_latent_buffer, loss_means_from_buff=False),
            num_classes=num_classes,
            *args, 
            **kwargs
        )
        
        self.norm = l2_latent_norm
        self.norm_lambda = norm_lambda
        self.alpha = alpha
        self.buff_on_same_device = buff_on_same_device

        self._means_once = False

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        loss = self._loss_f(latent, y)

        if(self.norm_lambda != 0.):
            norm = self.norm(latent, self.norm_lambda)
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
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(self._loss_f.classify(latent), y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        test_loss = self._loss_f(latent, y, train=False)
        self.log("test_loss", test_loss, on_step=True)

        classified_to_class = self._loss_f.classify(latent)
        self.test_acc(classified_to_class, y)
        self.log("test_acc", self.test_acc)

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
        if(self._robust_model_set):
            return 'CLModelWithIslands_' + type(self).__qualname__ + '_' + type(self.model.model).__qualname__
        return 'CLModelWithIslands_' + type(self).__qualname__ + '_' + type(self.model).__qualname__
        