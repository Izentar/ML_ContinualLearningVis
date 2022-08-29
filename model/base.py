from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.autograd.variable import Variable
import torchmetrics

from config.default import optim_Adam_config
from typing import Union
from abc import abstractmethod

class CLBase(LightningModule):
    def __init__(self, num_tasks, num_classes, dream_frequency:Union[int,list[int]]=1):
        super().__init__()

        self.train_acc = torchmetrics.Accuracy()
        self.train_acc_dream = torchmetrics.Accuracy()
        self.valid_accs = nn.ModuleList(
            [torchmetrics.Accuracy() for _ in range(num_tasks)]
        )
        self.test_acc = torchmetrics.Accuracy()
        self.num_classes = num_classes

        self.dream_frequency = dream_frequency if dream_frequency >= 1 else 1

    def training_step(self, batch, batch_idx):
        batch_normal = batch["normal"]
        loss_normal = self.training_step_normal(batch_normal)
        if "dream" not in batch:
            return loss_normal
        #TODO do we really need to compute dreams at each batch?
        #if (isinstance(self.dream_frequency, list) and batch_idx in self.dream_frequency) or \
        #    (isinstance(self.dream_frequency, int) and batch_idx % self.dream_frequency == 0):
        #    loss_dream = self.training_step_dream(batch["dream"])
        #    return loss_normal + loss_dream
        return loss_normal
    
    @abstractmethod
    def training_step_normal(self, batch):
        pass

    @abstractmethod
    def training_step_dream(self, batch):
        pass

    @abstractmethod
    def get_objective_target(self):
        """
            Returns the list of strings with layers names of the objective target.
            It should return the target layer name currently used in model.
        """
        raise Exception("Not implemented")

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=optim_Adam_config["lr"])



