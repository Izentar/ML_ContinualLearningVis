from model import base
import torch
from torch import nn, sigmoid
from torch.nn.functional import relu, cross_entropy, mse_loss
from torch.autograd.variable import Variable
from robustness import model_utils
from loss_function.chiLoss import ChiLoss, l2_latent_norm, ChiLossOneHot
from utils.data_manipulation import select_class_indices_tensor
from utils.cyclic_buffer import CyclicBufferByClass
from loss_function.chiLoss import ChiLossBase, DummyLoss
from model.SAE import SAE_CIFAR

from config.default import datasets, datasets_map

class SAE_standalone(base.CLBase):
    def __init__(self, num_classes, loss_f=None, reconstruction_loss_f=None, enable_robust=False, *args, **kwargs):
        super().__init__(num_classes=num_classes, *args, **kwargs)

        self.model = SAE_CIFAR(num_classes)
        self.enable_robust = enable_robust

        self.loss_f = loss_f if loss_f is not None else cross_entropy
        self.reconstruction_loss_f = reconstruction_loss_f if reconstruction_loss_f is not None else mse_loss

    def training_step_normal(self, batch):
        x, y = batch
        y_hat, y_reconstruction = self(x)
        loss_classification = self.loss_f(y_hat, y)
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
        loss_classification = self.loss_f(y_hat, y)
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
        val_loss = self.loss_f(y_hat, y)
        self.log("val_loss", val_loss)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(y_hat, y)
        self.log("valid_step_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        test_loss = self.loss_f(y_hat, y)
        self.log("test_loss", test_loss)
        self.test_acc(y_hat, y)
        self.log("test_step_acc", self.test_acc)

    def get_objective_target(self):
        return "model_fc"

    def get_root_objective_target(self): 
        if(self.enable_robust):
            return "model_model_" + self.model.model.get_root_name()
        return "model_" + self.model.get_root_name()

    def forward(self, *args):
        return self.model(*args)


class CLModel(base.CLBase):
    def __init__(
        self,
        model=None,
        loss_f=None,
        load_model:bool=False,
        robust_dataset=None,
        robust_dataset_name:str=None,
        robust_data_path:str=None,
        attack_kwargs=None,
        dreams_with_logits=False,
        resume_path=None,
        enable_robust=False,
        *args, 
        **kwargs
    ):
    
        super().__init__(load_model=load_model, *args, **kwargs)

        #if(load_model is None and (model is None or loss_f is None)):
        #    raise Exception('Model or loss function not provided. This can be omnited only when loading model.')

        self.attack_kwargs = attack_kwargs
        self.dreams_with_logits = dreams_with_logits
        self.enable_robust = enable_robust
        self.robust_data_path = robust_data_path
        self.robust_dataset_name = robust_dataset_name
        self.resume_path = resume_path
        if(enable_robust):
            robust_dataset = self._get_dataset_list(robust_dataset_name)[1](data_path=robust_data_path)
            if(robust_dataset_name is not None and robust_data_path is not None and
                robust_dataset is not None and attack_kwargs is not None):
                self.model = model_utils.make_and_restore_model(
                    arch=model, dataset=robust_dataset, resume_path=resume_path
                )[0]
            else:
                raise Exception('Robust selected but robust_dataset or attack_kwargs not provided.')
        else:
            self.model = model
            
        self.loss_f = loss_f if isinstance(loss_f, ChiLossBase) else DummyLoss(loss_f)
        print(f"INFO: Using loss {str(self.loss_f)}")

        self.save_hyperparameters(ignore=['model', 'loss_f', 'optim_manager', 
        'optimizer_construct_f', 'scheduler_construct_f', 'optimizer_restart_params_f'])

    def _get_dataset_list(name:str):
        if name is not None:
            cap = name.capitalize()
            if(cap in datasets_map):
                return datasets_map[cap]
        raise Exception(f"Unknown dataset {name}.")

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def call_loss(self, input, target, train, **kwargs):
        return self.loss_f(input, target, train)

    def training_step_normal(self, batch):
        x, y = batch

        if(self.enable_robust):
            model_out = self(
                x, target=y, **self.attack_kwargs
            )
        else:
            model_out = self(x)
        latent, model_out_dict = self.get_model_out_data(model_out)
        
        a = torch.abs(latent.detach()).sum().cpu().item()
        self.log("train_loss/latent_model_abs_sum", a)
        loss = self.process_losses_normal(
            x=x, 
            y=y, 
            latent=latent, 
            model_out_dict=model_out_dict,
            log_label="train_loss", 
        )
        self.log("train_loss/total", loss)
        self.train_acc(self.loss_f.classify(latent), y)
        self.log("train_step_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def process_losses_normal_reconstruction(self, x, loss, log_label, alpha=0.05, model_out_dict=None):
        if(model_out_dict is not None and 'image_reconstruction' in model_out_dict):
            loss_reconstruction = mse_loss(model_out_dict['image_reconstruction'], Variable(x))
            self.log(f"{log_label}/reconstuction_loss", loss_reconstruction)
            loss = loss * alpha + (1 - alpha) * loss_reconstruction
        return loss

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        loss = self.loss_f(latent, y)
        self.log(f"{log_label}/classification_loss", loss)

        loss = self.process_losses_normal_reconstruction(
            x=x, 
            loss=loss, 
            log_label=log_label, 
            model_out_dict=model_out_dict
        )
        return loss

    def training_step_dream(self, batch):
        if self.dreams_with_logits:
            x, logits, y = batch
        else:
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
        self.train_acc_dream(self.loss_f.classify(latent), y)
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
        val_loss = cross_entropy(self.loss_f.classify(latent), y)
        self.log("val_loss", val_loss)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(latent, y)
        self.log("valid_step_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        test_loss = cross_entropy(self.loss_f.classify(latent), y)
        self.log("test_loss", test_loss)
        self.test_acc(self.loss_f.classify(latent), y)
        self.log("test_step_acc", self.test_acc)

    def get_objective_target(self):
        if(self.enable_robust):
            return "model_model_" + self.model.model.get_objective_layer_name()
        return "model_" + self.model.get_objective_layer_name()

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
            return "model_model_" + self.model.model.get_root_name()
        return "model_" + self.model.get_root_name()

    def loss_to(self, device):
        return

    def get_obj_str_type(self) -> str:
        return 'CLModel_' + type(self.model).__qualname__

class CLModelIslandsTest(CLModel):
    def __init__(self, *args, hidden=10, num_classes=10, one_hot_means=None, size_per_class=40, **kwargs):
        kwargs.pop('loss_f', None)
        super().__init__(num_classes=num_classes, *args, loss_f=torch.nn.MSELoss(), **kwargs)
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=num_classes, dimensions=hidden, size_per_class=size_per_class)
        
        self.one_hot_means = one_hot_means
        self.loss_f = ChiLossOneHot(one_hot_means, self.cyclic_latent_buffer, loss_f=self.loss_f)

        self.valid_correct = 0
        self.valid_all = 0

    def call_loss(self, input, target, train, **kwargs):
        return self.loss_f(input, target, train)

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    def decode(self, target):
        return self.loss_f.decode(target)

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        loss = self.loss_f(latent, y)
        self.log(f"{log_label}/MSE_loss", loss)

        loss = self.process_losses_normal_reconstruction(
            x=x, 
            loss=loss, 
            log_label=log_label, 
            model_out_dict=model_out_dict
        )

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
        val_loss = self.loss_f(y_model, y)
        self.log("val_loss", val_loss)

        classified_to_class = self.loss_f.classify(y_model)
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
        test_loss = self.loss_f(y_model, y)
        self.log("test_loss", test_loss)

        classified_to_class = self.loss_f.classify(y_model)
        self.test_acc(classified_to_class, y)
        #self.valid_to_class(classified_to_class, y)
        self.log("test_acc", self.test_acc)

    def get_buffer(self):
        return self.cyclic_latent_buffer

    def get_obj_str_type(self) -> str:
        if(self.enable_robust):
            return 'CLModelIslandsTest_' + type(self).__qualname__ + '_' + type(self.model.model).__qualname__
        return 'CLModelIslandsTest_' + type(self).__qualname__ + '_' + type(self.model).__qualname__

class CLModelWithIslands(CLModel):
    def __init__(
            self, 
            hidden, 
            cyclic_latent_buffer_size_per_class, 
            num_classes, 
            *args, 
            buff_on_same_device=False,  
            alpha=0.0, 
            norm_lambda=0.001, 
            sigma=0.2, 
            rho=1., 
            **kwargs
        ):
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=num_classes, dimensions=hidden, size_per_class=cyclic_latent_buffer_size_per_class)
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

    def call_loss(self, input, target, train=True, **kwargs):
        return self.loss_f(input, target, train=train)

    def loss_to(self, device):
        self.loss_f.to(device)

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        loss = self.loss_f(latent, y)

        for k, v in self.loss_f.to_log.items():
            self.log(k, v)

        if(self.norm_lambda != 0.):
            norm = self.norm(latent, self.norm_lambda)
            self.log(f"{log_label}/norm", norm)
            loss += norm
        self.log(f"{log_label}/island", loss)

        loss = self.process_losses_normal_reconstruction(
            x=x, 
            loss=loss, 
            log_label=log_label, 
            model_out_dict=model_out_dict
        )

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
        val_loss = self.loss_f(latent, y, train=False)
        self.log("val_loss", val_loss)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(self.loss_f.classify(latent), y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        model_out = self(x)
        latent, _ = self.get_model_out_data(model_out)
        test_loss = self.loss_f(latent, y, train=False)
        self.log("test_loss", test_loss)

        self.test_acc(self.loss_f.classify(latent), y)
        self.log("test_acc", self.test_acc)


    def training_step(self, batch, batch_idx):
        if(self.cyclic_latent_buffer is not None and self.buff_on_same_device):
            self.cyclic_latent_buffer.to(self.device)
        return super().training_step(batch=batch, batch_idx=batch_idx)
        
    def get_buffer(self):
        return self.cyclic_latent_buffer

    def get_obj_str_type(self) -> str:
        return 'CLModelWithIslands_' + type(self).__qualname__ + '_' + type(self.model.model).__qualname__