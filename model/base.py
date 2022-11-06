from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.autograd.variable import Variable
import torchmetrics

from config.default import optim_Adam_config
from typing import Union
from abc import abstractmethod

class CLBase(LightningModule):
    def __init__(
        self, 
        num_tasks, 
        num_classes, 
        dream_frequency:Union[int,list[int]]=1, 
        data_passer: dict=None, 
        only_dream_batch=False,
        optimizer_construct_f=None,
        scheduler_construct_f=None,
        scheduler_steps=None,
        *args, 
        **kwargs
    ):
        '''
            optimizer_construct_f - function with signature fun(parameters)
            scheduler_construct_f - function with signature fun(optim)
        '''
        super().__init__()
        # ignore *args, **kwargs

        assert not (scheduler_construct_f is not None and optimizer_construct_f is None), "Scheduler should have optimizer"
        assert not (scheduler_steps is not None and scheduler_construct_f is None), "Scheduler steps should have scheduler"

        self.train_acc = torchmetrics.Accuracy()
        self.train_acc_dream = torchmetrics.Accuracy()
        self.valid_accs = nn.ModuleList(
            [torchmetrics.Accuracy() for _ in range(num_tasks)]
        )
        self.test_acc = torchmetrics.Accuracy()
        self.num_classes = num_classes

        self.dream_frequency = dream_frequency if dream_frequency >= 1 else 1
        self.data_passer = data_passer
        self.only_dream_batch = only_dream_batch
        self.optimizer_construct_f = optimizer_construct_f
        self.scheduler_construct_f = scheduler_construct_f
        self.scheduler = None

        self.scheduler_steps = scheduler_steps if scheduler_steps is not None else (None, )

    def _inner_training_step(self, batch, batch_idx):
        batch_normal = batch["normal"]
        loss_normal = self.training_step_normal(batch_normal)
        if "dream" not in batch:
            return loss_normal
        if self.only_dream_batch:
            return self.training_step_dream(batch["dream"])
        #TODO Czy naprawdę potrzebujemy uczyć się na snach w każdym batchu? W każdym z nich będą te same obrazki zawsze.
        #TODO może powinniśmy wymieszać obrazki ze snów ze zwykłymi obrazkami?
        if (isinstance(self.dream_frequency, list) and batch_idx in self.dream_frequency) or \
            (isinstance(self.dream_frequency, int) and batch_idx % self.dream_frequency == 0):
            loss_dream = self.training_step_dream(batch["dream"])
            return loss_normal + loss_dream
        return loss_normal
    
    def training_step(self, batch, batch_idx):
        return self._inner_training_step(batch, batch_idx)

    def get_model_out_data(self, model_out):
        """
            Return latent and model output dictionary if exist else None
        """
        model_out_dict = None
        latent = model_out
        if(isinstance(model_out, tuple)):
            latent = model_out[0]
            model_out_dict = model_out[1]
        return latent, model_out_dict

    @abstractmethod
    def training_step_normal(self, batch):
        pass

    @abstractmethod
    def training_step_dream(self, batch):
        pass

    @abstractmethod
    def call_loss(self, input, target):
        pass

    @abstractmethod
    def get_objective_target(self):
        """
            It should return the target layer name currently used in model.
        """
        raise Exception("Not implemented")

    @abstractmethod
    def get_root_objective_target(self): 
        """
            Return root string with a hard space at the end.
        """
        pass

    @abstractmethod
    def get_obj_str_type(self) -> str:
        pass

    def configure_optimizers(self):
        if(self.optimizer_construct_f is None):
            optim = torch.optim.Adam(self.parameters(), lr=optim_Adam_config["lr"])
        else:
            optim = self.optimizer_construct_f(self.parameters())
        
        if(self.scheduler_construct_f is not None):
            self.scheduler = self.scheduler_construct_f(optim)
        return optim

    def on_train_batch_start(self, batch, batch_idx):
        return self.data_passer['model_train_end_f']()

        #if(self.data_passer is not None and 
        #string in self.data_passer and 
        #self.data_passer[string] >= 1 and 
        #self.dream_only_once and 
        #batch_idx >= 0):
        #    return -1

    # training_epoch_end
    def training_epoch_end(self, output):
        print(f"debug current_task_loop: {self.data_passer['current_task_loop']}, self.scheduler_steps {self.scheduler_steps}")
        if(self.scheduler is not None):
            if(self.current_epoch in self.scheduler_steps):
                self.scheduler.step()
                print(f"Changed learning rate to: {self.scheduler._last_lr}, debug current_task_loop: {self.data_passer['current_task_loop']}, self.scheduler_steps {self.scheduler_steps}")