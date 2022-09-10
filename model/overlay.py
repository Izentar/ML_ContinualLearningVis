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

        y_latent = self(
            x, target=y, make_adv=self.train_normal_robustly, **self.attack_kwargs
        )
        a = torch.abs(y_latent.detach()).sum().cpu().item()
        self.log("train_loss/latent_model_abs_sum", a)
        loss = self.process_losses_normal(
            x=x, 
            y=y, 
            y_latent=y_latent, 
            label="train_loss", 
        )
        self.log("train_loss/total", loss)
        self.train_acc(self.loss_f.classify(y_latent), y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def process_losses_normal(self, x, y, y_latent, label):
        loss = self.loss_f.classify(y_latent)
        self.log(f"{label}/classification", loss)
        return loss

    def training_step_dream(self, batch):
        if self.dreams_with_logits:
            x, logits, y = batch
        else:
            x, y = batch

        y_latent = self(
            x, target=y, make_adv=self.train_dreams_robustly, **self.attack_kwargs
        )
        loss = self.process_losses_dreams(
            x, y, y_latent, "train_loss_dream",
        )
        self.log("train_loss_dream/total", loss)
        self.train_acc_dream(self.loss_f.classify(y_latent), y)
        self.log("train_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)
        return loss

    def process_losses_dreams(self, x, y, y_latent, label):
        return self.process_losses_normal(x, y, y_latent, label)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_latent = self(x)
        val_loss = cross_entropy(self.loss_f.classify(y_latent), y)
        self.log("val_loss", val_loss)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(y_latent, y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_latent = self(x)
        test_loss = cross_entropy(self.loss_f.classify(y_latent), y)
        self.log("test_loss", test_loss)
        self.test_acc(self.loss_f.classify(y_latent), y)
        self.log("test_acc", self.test_acc)

    def get_objective_target(self):
        return "model_model_" + self.model.model.get_objective_layer_name()

class CLModelWithReconstruction(CLModel):
    def __init__(self, *args, dreams_reconstruction_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dreams_reconstruction_loss = dreams_reconstruction_loss

    def process_losses_normal(self, x, y, y_latent, label):
        loss_latent = mse_loss(y_latent, Variable(x))
        alpha = 0.05
        loss = alpha * loss_classification + (1 - alpha) * loss_latent
        self.log(f"{label}/latent", loss_latent)
        return loss

    def call_loss(self, input, target, **kwargs):
        return mse_loss(input, target)

    def process_losses_dreams(self, x, y, y_latent, label):
        if self.dreams_reconstruction_loss:
            return self.process_losses_normal(x, y, y_latent, label)
        return super().process_losses_normal(x, y, y_latent, label)

class CLModelIslandsTest(CLModel):
    def __init__(self, *args, hidden=10, one_hot_means=None, only_one_hot=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=10, dimensions=hidden, size_per_class=40)
        self.loss_f = torch.nn.MSELoss() #torch.nn.CrossEntropyLoss()
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

    def process_losses_normal(self, x, y, y_latent, label):
        y_decoded = self.decode(y)
        loss = self.loss_f(y_latent, y_decoded.float())
        self.log(f"{label}/cross_entropy_loss", loss)

        selected_class = 0
        uniq = torch.unique(y)
        cl_batch = None
        for u in uniq:
            if(u.item() == selected_class):
                cl_batch = y_latent[y == u]
                break
        if(cl_batch is not None):
            cl_batch = torch.mean(cl_batch, dim=0)
            for idx, m in enumerate(cl_batch):
                self.log(f"val_cl0_dim{idx}", m)

        return loss

    def process_losses_dreams(self, x, y, y_latent, label):
        return self.process_losses_normal(x, y, y_latent, label)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y = self.decode(y)
        y_cl = self(x)
        val_loss = self.loss_f(y_cl, y)
        self.log("val_loss", val_loss)

        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(y_cl, y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y = self.decode(y)
        y_cl = self(x)
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

    def process_losses_normal(self, x, y, y_latent, label):
        loss = self.loss_f(y_latent, y)

        for k, v in self.loss_f.to_log.items():
            self.log(k, v)

        if(self.norm_lambd != 0.):
            norm = self.norm(y_latent, self.norm_lambd)
            self.log(f"{label}/norm", loss)
            loss += norm
        self.log(f"{label}/island", loss)

        return loss

    def process_losses_dreams(self, x, y, y_latent, label):
        return self.process_losses_normal(x, y, y_latent, label)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_latent = self(x)
        val_loss = self.loss_f(y_latent, y, train=False)
        self.log("val_loss", val_loss)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(self.loss_f.classify(y_latent), y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_latent = self(x)
        test_loss = self.loss_f(y_latent, y, train=False)
        self.log("test_loss", test_loss)

        self.test_acc(self.loss_f.classify(y_latent), y)
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