from model import base
import torch
from torch import nn, sigmoid
from torch.nn.functional import relu, cross_entropy, mse_loss
from torch.autograd.variable import Variable
from robustness import model_utils
from loss_function.chiLoss import ChiLoss, l2_latent_norm
from utils.data_manipulation import select_class_indices_tensor
from utils.cyclic_buffer import CyclicBufferByClass

class CLModel(base.CLBase):
    def __init__(
        self,
        model,
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
        self.source_model = model

    def forward(self, *args, make_adv=False, with_image=False, **kwargs):
        return self.model(*args, make_adv=make_adv, with_image=with_image, **kwargs)

    def training_step_normal(self, batch):
        x, y = batch

        def loss_fn(y_hat):
            return cross_entropy(y_hat, y)

        y_hat, *y_auxiliary = self(
            x, target=y, make_adv=self.train_normal_robustly, **self.attack_kwargs
        )
        a = torch.abs(y_auxiliary[0].detach()).sum().cpu().item()
        self.log("train_loss/xe_latent", a)
        loss = self.process_losses_normal(
            x=x, 
            y=y, 
            y_hat=y_hat, 
            y_auxiliary=y_auxiliary, 
            label="train_loss", 
            loss_fn=loss_fn,
        )
        self.log("train_loss/total", loss)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def process_losses_normal(self, x, y, y_hat, y_auxiliary, label, loss_fn):
        loss = loss_fn(y_hat)
        self.log(f"{label}/classification", loss)
        return loss

    def training_step_dream(self, batch):
        if self.dreams_with_logits:
            x, logits, y = batch

            def loss_fn(y_hat):
                return 0.3 * mse_loss(y_hat, logits)

        else:
            x, y = batch

            def loss_fn(y_hat):
                return cross_entropy(y_hat, y)

        y_hat, *y_auxiliary = self(
            x, target=y, make_adv=self.train_dreams_robustly, **self.attack_kwargs
        )
        loss = self.process_losses_dreams(
            x, y, y_hat, y_auxiliary, "train_loss_dream", loss_fn=loss_fn
        )
        self.log("train_loss_dream/total", loss)
        self.train_acc_dream(y_hat, y)
        self.log("train_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)
        return loss

    def process_losses_dreams(self, x, y, y_hat, y_auxiliary, label, loss_fn):
        return self.process_losses_normal(x, y, y_hat, y_auxiliary, label, loss_fn)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_hat = self(x)[0]
        val_loss = cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(y_hat, y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)[0]
        test_loss = cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc)

    def get_objective_target(self):
        return "model_model_fc"

class CLModelWithReconstruction(CLModel):
    def __init__(self, *args, dreams_reconstruction_loss=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.dreams_reconstruction_loss = dreams_reconstruction_loss

    def process_losses_normal(self, x, y, y_hat, y_auxiliary, label, loss_fn):
        loss_classification = loss_fn(y_hat)
        loss_reconstruction = mse_loss(y_auxiliary[0], Variable(x))
        alpha = 0.05
        loss = alpha * loss_classification + (1 - alpha) * loss_reconstruction
        self.log(f"{label}/classification", loss_classification)
        self.log(f"{label}/reconstrucion", loss_reconstruction)
        return loss

    def process_losses_dreams(self, x, y, y_hat, y_auxiliary, label, loss_fn):
        if self.dreams_reconstruction_loss:
            return self.process_losses_normal(x, y, y_hat, y_auxiliary, label, loss_fn)
        return super().process_losses_normal(x, y, y_hat, y_auxiliary, label, loss_fn)

class CLModelWithIslands(CLModel):
    def __init__(self, *args, hidden=3, buff_on_same_device=False, islands=False, alpha=0.0, norm_lambd=0.001, sigma=0.2, rho=1., **kwargs):
        super().__init__(*args, **kwargs)
        self.islands = islands
        #self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=kwargs.get('num_classes'), dimensions=hidden, size_per_class=40)
        self.cyclic_latent_buffer = None
        self.island_loss = ChiLoss(sigma=sigma, rho=rho, cyclic_buffer=self.cyclic_latent_buffer)
        self.norm = l2_latent_norm
        self.norm_lambd = norm_lambd
        self.alpha = alpha
        self.buff_on_same_device = buff_on_same_device

    def process_losses_normal(self, x, y, y_hat, y_auxiliary, label, loss_fn):
        #loss_classification = loss_fn(y_hat)
        norm = self.norm(y_auxiliary[0], self.norm_lambd)
        loss_island = self.island_loss(y_auxiliary[0], y)

        loss = loss_island + norm
        self.log(f"{label}/island", loss_island)
        self.log(f"{label}/l2-regularization", norm)
        #self.log(f"{label}/distance_weight", self.island_loss.distance_weight)
        self.log(f"{label}/positive_loss", self.island_loss.positive_loss.detach().sum())
        self.log(f"{label}/negative_loss", self.island_loss.negative_loss.detach().sum())
        #self.log(f"{label}/input_sum", self.island_loss.input_sum)
        return loss

    def process_losses_dreams(self, x, y, y_hat, y_auxiliary, label, loss_fn):
        if self.islands:
            return self.process_losses_normal(x, y, y_hat, y_auxiliary, label, loss_fn)
        return super().process_losses_normal(x, y, y_hat, y_auxiliary, label, loss_fn)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_latent = self(x)[1]
        val_loss = self.island_loss(y_latent, y, train=False)
        self.log("val_loss", val_loss)

        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(self.island_loss.classify(y_latent), y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_latent = self(x)[1]
        test_loss = self.island_loss(y_latent, y, train=False)
        self.log("test_loss", test_loss)

        self.test_acc(self.island_loss.classify(y_latent), y)
        self.log("test_acc", self.test_acc)

    def _calculate_cov(self, inp, target):
        target_unique, target_count = torch.unique(target, return_counts=True)
        max_idx = torch.argmax(target_count)
        main_target_class = target_unique[max_idx]
        
        select_class_indices_tensor(main_target, target)
        cl_indices = torch.isin(new_buffer_target, cl)
        cl_indices_list = torch.where(cl_indices)[0]

        cov_matrix = torch.cov(inp)
        pass

    def training_step(self, batch, batch_idx):
        if(self.cyclic_latent_buffer is not None and self.buff_on_same_device):
            self.cyclic_latent_buffer.to(self.device)
        return super().training_step(batch=batch, batch_idx=batch_idx)
        