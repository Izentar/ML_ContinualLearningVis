from model import base
import torch
from torch import nn, sigmoid
from torch.nn.functional import relu, cross_entropy, mse_loss
from torch.autograd.variable import Variable
from robustness import model_utils
from loss_function.chiLoss import ChiLoss, l2_latent_norm, ChiLossOneHot
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

        y_latent = self(
            x, target=y, make_adv=self.train_normal_robustly, **self.attack_kwargs
        )
        a = torch.abs(y_latent.detach()).sum().cpu().item()
        self.log("train_loss/xe_latent", a)
        loss = self.process_losses_normal(
            x=x, 
            y=y, 
            y_latent=y_latent, 
            label="train_loss", 
        )
        self.log("train_loss/total", loss)
        #self.train_acc(y_hat, y)
        #self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    #def process_losses_normal(self, x, y, y_latent, label):
    #    loss = loss_fn(y_hat)
    #    self.log(f"{label}/classification", loss)
    #    return loss

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
        self.train_acc_dream(y_hat, y)
        self.log("train_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)
        return loss

    def process_losses_dreams(self, x, y, y_latent, label):
        return self.process_losses_normal(x, y, y_latent, label)

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

    def process_losses_normal(self, x, y, y_latent, label):
        loss_latent = mse_loss(y_latent, Variable(x))
        alpha = 0.05
        loss = alpha * loss_classification + (1 - alpha) * loss_latent
        self.log(f"{label}/latent", loss_latent)
        return loss

    def process_losses_dreams(self, x, y, y_latent, label):
        if self.dreams_reconstruction_loss:
            return self.process_losses_normal(x, y, y_latent, label)
        return super().process_losses_normal(x, y, y_latent, label)

class CLModelWithIslands(CLModel):
    def __init__(self, *args, hidden=3, buff_on_same_device=False, islands=False, alpha=0.0, norm_lambd=0.001, sigma=0.2, rho=1., one_hot_means=None, only_one_hot=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.islands = islands
        #self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=kwargs.get('num_classes'), dimensions=hidden, size_per_class=40)
        self.cyclic_latent_buffer = None
        #self.island_loss = ChiLoss(sigma=sigma, rho=rho, cyclic_buffer=self.cyclic_latent_buffer)

        self.cyclic_latent_buffer = CyclicBufferByClass(num_classes=10, dimensions=7, size_per_class=40)
        self.island_loss = ChiLossOneHot(cyclic_latent_buffer=self.cyclic_latent_buffer, sigma=sigma, rho=rho, one_hot_means=one_hot_means, only_one_hot=only_one_hot)
        
        self.norm = l2_latent_norm
        self.norm_lambd = norm_lambd
        self.alpha = alpha
        self.buff_on_same_device = buff_on_same_device

    def process_losses_normal(self, x, y, y_latent, label):
        norm = self.norm(y_latent, self.norm_lambd)
        loss_island = self.island_loss(y_latent, y)

        loss = loss_island + norm
        self.log(f"{label}/island", loss_island)
        #self.log(f"{label}/island", loss_island)

        #self.log(f"{label}/l2-regularization", norm)
        #self.log(f"{label}/distance_weight", self.island_loss.distance_weight)
        #self.log(f"{label}/positive_loss", self.island_loss.positive_loss.detach().sum())
        #self.log(f"{label}/negative_loss", self.island_loss.negative_loss.detach().sum())
        #self.log(f"{label}/input_sum", self.island_loss.input_sum)
        return loss

    def process_losses_dreams(self, x, y, y_latent, label):
        if self.islands:
            return self.process_losses_normal(x, y, y_latent, label)
        return super().process_losses_normal(x, y, y_latent, label)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y = batch
        y_latent = self(x)
        val_loss = self.island_loss(y_latent, y, train=False)
        self.log("val_loss", val_loss)
        pdist = torch.nn.PairwiseDistance(p=2)

        means = self.cyclic_latent_buffer.mean()
        #for i in range(10):
        #    self.log(f"val_means_{i}", pdist(means[i], torch.zeros((7,), device='cpu')))

        for i, m in enumerate(means[0]):
            self.log(f"val_means_cl0_dim{i}", m)

        #valid_acc = self.valid_accs[dataloader_idx]
        #valid_acc(self.island_loss.classify(y_latent), y)
        #self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_latent = self(x)
        test_loss = self.island_loss(y_latent, y, train=False)
        self.log("test_loss", test_loss)

        #self.test_acc(self.island_loss.classify(y_latent), y)
        #self.log("test_acc", self.test_acc)

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
        