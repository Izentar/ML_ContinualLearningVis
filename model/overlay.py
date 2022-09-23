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

class CLModel(base.CLBase):
    def __init__(
        self,
        model,
        loss_f,
        robust_dataset,
        attack_kwargs,
        dreams_with_logits=False,
        train_normal_robustly=False,
        train_dreams_robustly=False,
        resume_path=None,
        *args, 
        **kwargs
    ):
        super().__init__(*args, **kwargs)

        self.attack_kwargs = attack_kwargs
        self.dreams_with_logits = dreams_with_logits
        self.train_normal_robustly = train_normal_robustly
        self.train_dreams_robustly = train_dreams_robustly
        self.model = model_utils.make_and_restore_model(
            arch=model, dataset=robust_dataset, resume_path=resume_path
        )[0]

        if(isinstance(loss_f, ChiLossBase)):
            self.loss_f = loss_f
        else:
            self.loss_f = DummyLoss(loss_f)

    def forward(self, *args, make_adv=False, with_image=False, **kwargs):
        return self.model(*args, make_adv=make_adv, with_image=with_image, **kwargs)

    def call_loss(self, input, target, **kwargs):
        return cross_entropy(input, target)

    def training_step_normal(self, batch):
        x, y = batch

        model_out = self(
            x, target=y, make_adv=self.train_normal_robustly, **self.attack_kwargs
        )
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

        model_out = self(
            x, target=y, make_adv=self.train_dreams_robustly, **self.attack_kwargs
        )
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
        return "model_model_" + self.model.model.get_objective_layer_name()

    def loss_to(self, device):
        return

class CLModelIslandsTest(CLModel):
    def __init__(self, *args, hidden=10, one_hot_means=None, only_one_hot=False, **kwargs):
        kwargs.pop('loss_f', None)
        super().__init__(*args, loss_f=torch.nn.MSELoss(), **kwargs)
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=10, dimensions=hidden, size_per_class=40)
        self.one_hot_means = one_hot_means

    def decode(self, target):
        one_hot = []
        for t in target:
            one_hot.append(self.one_hot_means[t.item()].to(t.device))
        return torch.stack(one_hot, 0)

    def call_loss(self, input, target, **kwargs):
        return self.loss_f(input, self.decode(target))

    def forward(self, *args, make_adv=False, with_image=False, **kwargs):
        return self.model(*args, make_adv=make_adv, with_image=with_image, **kwargs)

    def process_losses_normal(self, x, y, latent, log_label, model_out_dict=None):
        y_decoded = self.decode(y)
        loss = self.loss_f(latent, y_decoded.float())
        self.log(f"{label}/MSE_loss", loss)

        loss = self.process_losses_normal_reconstruction(
            x=x, 
            loss=loss, 
            log_label=log_label, 
            model_out_dict=model_out_dict
        )

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
        y = self.decode(y)
        model_out = self(x)
        y_cl, _ = self.get_model_out_data(model_out)
        val_loss = self.loss_f(y_cl, y)
        self.log("val_loss", val_loss)

        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(y_cl, y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = self.decode(y)
        model_out = self(x)
        y_cl, _ = self.get_model_out_data(model_out)
        test_loss = self.loss_f(y_cl, y)
        self.log("test_loss", test_loss)
        self.test_acc(y_cl, y)
        self.log("test_acc", self.test_acc)

class CLModelWithIslands(CLModel):
    def __init__(
            self, 
            hidden, 
            cyclic_latent_buffer_size_per_class, 
            num_classes, 
            *args, 
            buff_on_same_device=False, 
            islands=False, 
            alpha=0.0, 
            norm_lambd=0.001, 
            sigma=0.2, 
            rho=1., 
            one_hot_means=None, 
            only_one_hot=False, 
            **kwargs
        ):
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=num_classes, dimensions=hidden, size_per_class=cyclic_latent_buffer_size_per_class)
        kwargs.pop('loss_f', None)
        super().__init__(
            loss_f=ChiLoss(sigma=sigma, rho=rho, cyclic_latent_buffer=self.cyclic_latent_buffer, loss_means_from_buff=False),
            #loss_f = ChiLossOneHot(cyclic_latent_buffer=self.cyclic_latent_buffer, sigma=sigma, rho=rho, one_hot_means=one_hot_means, only_one_hot=only_one_hot)
            num_classes=num_classes,
            *args, 
            **kwargs
        )
        
        self.norm = l2_latent_norm
        self.norm_lambd = norm_lambd
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

        if(self.norm_lambd != 0.):
            norm = self.norm(latent, self.norm_lambd)
            self.log(f"{label}/norm", norm)
            loss += norm
        self.log(f"{label}/island", loss)

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

    def _calculate_cov(self, inp, target):
        target_unique, target_count = torch.unique(target, return_counts=True)
        max_idx = torch.argmax(target_count)
        main_target_class = target_unique[max_idx]
        
        select_class_indices_tensor(main_target, target)
        cl_indices = torch.isin(new_buffer_target, cl)
        cl_indices_list = torch.where(cl_indices)[0]

        cov_matrix = torch.cov(inp)

    def training_step(self, batch, batch_idx):
        if(self.cyclic_latent_buffer is not None and self.buff_on_same_device):
            self.cyclic_latent_buffer.to(self.device)
        return super().training_step(batch=batch, batch_idx=batch_idx)
        
    def get_buffer(self):
        return self.cyclic_latent_buffer