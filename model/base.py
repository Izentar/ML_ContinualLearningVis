from pytorch_lightning import LightningModule
import torch
from torch import nn
from torch.autograd.variable import Variable
import torchmetrics

from config.default import optim_Adam_config
from typing import Union, Sequence
from abc import abstractmethod
from utils.functional.model_optimizer import ModelOptimizerManager

class CLBase(LightningModule):
    def __init__(
        self, 
        num_tasks:int, 
        num_classes:int, 
        load_model:str=None,
        dream_frequency:Union[int,list[int]]=1, 
        data_passer:dict=None, 
        train_only_dream_batch:bool=False,
        optimizer_construct_type:str=None,
        scheduler_construct_type:str=None,
        optimizer_restart_params_type:str=None,
        optimizer_params:dict=None,
        scheduler_steps:Sequence[int]=None,
        swap_datasets:bool=False,
        *args, 
        **kwargs
    ):
        '''
            optimizer_construct_type - function with signature fun(parameters)
            scheduler_construct_type - function with signature fun(optim)
            optimizer_restart_params_type - function with signature fun(optimizer)
        '''
        super().__init__()
        # ignore *args, **kwargs

        #if(load_model is None and (num_tasks is None or num_classes is None)):
        #    raise Exception('Number of tasks or number of classes not provided. This can be omnited only when loading model.')

        # needed for pytorch lightning metadata save framework
        self.optimizer_construct_type = optimizer_construct_type
        self.scheduler_construct_type = scheduler_construct_type
        self.optimizer_restart_params_type = optimizer_restart_params_type
        self.optimizer_params = optimizer_params
        self.load_model = load_model

        self.optim_manager = ModelOptimizerManager(
            optimizer_type=optimizer_construct_type,
            scheduler_type=scheduler_construct_type,
            reset_optim_type=optimizer_restart_params_type,
        )

        assert not (scheduler_construct_type is not None and optimizer_construct_type is None), "Scheduler should have optimizer"
        if(scheduler_steps is not None and scheduler_construct_type is None):
            print('WARNIGN: Using "scheduler_steps" but scheduler is not selected.')

        self.train_acc = torchmetrics.Accuracy()
        self.train_acc_dream = torchmetrics.Accuracy()
        self.valid_accs = nn.ModuleList(
            [torchmetrics.Accuracy() for _ in range(num_tasks)]
        )
        self.test_acc = torchmetrics.Accuracy()
        self.num_classes = num_classes
        self.num_tasks = num_tasks

        self.dream_frequency = dream_frequency if dream_frequency >= 1 else 1
        print(f"Set dream frequency to: {self.dream_frequency}")
        self.data_passer = data_passer
        self.train_only_dream_batch = train_only_dream_batch
        self.optimizer_construct_f = self.optim_manager.get_optimizer(**optimizer_params)
        self.scheduler_construct_f = self.optim_manager.get_scheduler(**optimizer_params)
        self.optimizer_restart_params_f = self.optim_manager.get_reset_optimizer_f(**optimizer_params)

        self.scheduler = None
        self.optimizer = None

        self.scheduler_steps = scheduler_steps if scheduler_steps is not None else (None, )
        self.swap_datasets = swap_datasets

        if(self.swap_datasets):
            print(f"INFO: Model overlay in swap_datasets mode.")

    def _inner_training_step(self, batch, batch_idx):
        if "dream" not in batch:
            return self.training_step_normal(batch["normal"])
        if self.train_only_dream_batch:
            return self.training_step_dream(batch["dream"])

        loss_normal = self.training_step_normal(batch["normal"])
        #TODO Czy naprawdę potrzebujemy uczyć się na snach w każdym batchu? W każdym z nich będą te same obrazki zawsze.
        #TODO może powinniśmy wymieszać obrazki ze snów ze zwykłymi obrazkami?
        if (isinstance(self.dream_frequency, list) and batch_idx in self.dream_frequency) or \
            (isinstance(self.dream_frequency, int) and batch_idx % self.dream_frequency == 0):
            loss_dream = self.training_step_dream(batch["dream"])
            return loss_normal + loss_dream
        return loss_normal

    def _inner_training_step_swap_datasets(self, batch, batch_idx):
        if("dream" in batch):
            return self.training_step_dream(batch["dream"])
        elif("normal" in batch):
            return self.training_step_normal(batch["normal"])
        else:
            raise Exception(f'No batch found. Batch keys: {batch.keys()}')

    
    def training_step(self, batch, batch_idx):
        if(self.swap_datasets):
            loss = self._inner_training_step_swap_datasets(batch, batch_idx)
        else:
            loss = self._inner_training_step(batch, batch_idx)
        return loss

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

    @abstractmethod
    def get_objective_layer(self):
        pass

    @abstractmethod
    def get_objective_layer_output_shape(self):
        pass

    def configure_optimizers(self):
        if(self.optimizer_construct_f is None):
            optim = torch.optim.Adam(self.parameters(), lr=optim_Adam_config["lr"])
        else:
            optim = self.optimizer_construct_f(self.parameters())
        
        if(self.scheduler_construct_f is not None):
            self.scheduler = self.scheduler_construct_f(optim)
        self.optimizer = optim
        return optim

    def on_train_batch_start(self, batch, batch_idx):
        return self.data_passer['model_train_end_f']

        #if(self.data_passer is not None and 
        #string in self.data_passer and 
        #self.data_passer[string] >= 1 and 
        #self.dream_only_once and 
        #batch_idx >= 0):
        #    return -1

    # training_epoch_end
    def training_epoch_end(self, output):
        # OK
        #print(f"debug current_task_loop: {self.data_passer['current_task_loop']}, self.current_epoch {self.current_epoch}")
        #print(f"self.data_passer['epoch_per_task'] {self.data_passer['epoch_per_task']}, self.scheduler_steps {self.scheduler_steps}")
        if(self.scheduler is not None):
            if(self.current_epoch >= self.data_passer['epoch_per_task'] - 1):
                self.optimizer_restart_params_f(self.optimizer)
                self.scheduler = self.scheduler_construct_f(self.optimizer)
                print(f'Scheduler restarted at epoch {self.current_epoch} end. Learning rate: {self.scheduler._last_lr}')
                return
            if(self.current_epoch in self.scheduler_steps):
                self.scheduler.step()
                print(f"Changed learning rate to: {self.scheduler._last_lr}")
                #print(f"debug current_task_loop: {self.data_passer['current_task_loop']}, self.scheduler_steps {self.scheduler_steps}")