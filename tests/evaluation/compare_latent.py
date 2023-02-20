import torch
from dataset import dream_sets
from lucent.optvis import param
import wandb
from torch.optim.lr_scheduler import StepLR
from tests.evaluation.utils import CustomDreamDataModule
from utils.functional import dream_objective
from loss_function.chiLoss import ChiLossBase
import numpy as np
from utils.functional.target_processing import target_processing_latent_binary_classification


class CompareLatent():
    """
        Compare latent points.
        Sample point in latent space, then do dreaming using this point.
        After that compare the point from generated image to the previous sampled point.
        This is done by creating custom objective function that compares main point to the sampled point 
        using provided loss function (default MSELoss).
    """
    def __init__(self):
        self.scheduler = None
        self.logged_main_point = [0]
        self.point_from_model = []
        self.custom_loss_f = torch.nn.MSELoss() 

    def param_f_image(image_size, dreaming_batch_size, **kwargs):
        def param_f():
            # uses 2D Fourier coefficients
            # sd - scale of the random numbers [0, 1)
            return param.image(
                image_size, batch=dreaming_batch_size, sd=0.4, #fft=False
            )
        return param_f

    def wrapper_select_dream_tasks_f(used_class):
        def select_dream_tasks_f(tasks, task_index):
            return set(used_class)
        return select_dream_tasks_f

    def target_processing_decorator(self, f):
        def wrapper(target, model, *args, **kwargs):
            point = f(target, model, *args, **kwargs)
            self.logged_main_point[0] = point
            return point
        return wrapper

    def get_dream_optim(self):
        def inner(params):
            self.optimizer = torch.optim.Adam(params, lr=5e-3)
            #self.optimizer = torch.optim.SGD(params, lr=0.1, momentum=0.9)
            self.scheduler = StepLR(self.optimizer, step_size=1, gamma=0.1)
            return self.optimizer
        return inner

    def scheduler_step(self):
        self.scheduler.step()
        print('Scheduler step', self.scheduler._last_lr)

    def __call__(
            self, 
            model, 
            used_class, 
            logger, 
            dream_transform, 
            target_processing_f=None, 
            loss_f=None,
            enable_scheduler=True, 
            scheduler_steps=(1024*3, 1024*4, 1024*5),
            dream_threshold=(1024*6,),
            loss_obj_step_sample=False,
            disable_transforms=True,
        ):
        label = 'compare_latent'

        if(isinstance(loss_f, ChiLossBase)):
            raise Exception(f'ChiLoss cannot be used as loss function. Use default MSELoss.')

        custom_target_processing_f = self.target_processing_decorator(
            target_processing_f
        ) if target_processing_f is not None else self.target_processing_decorator(
            target_processing_latent_binary_classification
        )

        self.custom_loss_f = torch.nn.MSELoss() if loss_f is None else loss_f
        if(loss_obj_step_sample):
            objective_fun = dream_objective.dream_objective_latent_step_sample_normal_creator(
                loss_f=self.custom_loss_f,
                std_scale=0.1,
                latent_saver=self.point_from_model,
                label=label,
                logger=logger,
            )
        else:
            objective_fun = dream_objective.dream_objective_latent_lossf_compare_creator(
                loss_f=self.custom_loss_f,
                latent_saver=self.point_from_model,
                label=label,
                logger=logger,
            )

        dream_module = CustomDreamDataModule(
            train_tasks_split=[[used_class]],
            select_dream_tasks_f=CompareLatent.wrapper_select_dream_tasks_f(used_class),
            dream_objective_f=objective_fun,
            target_processing_f=custom_target_processing_f,
            dreams_per_target=1,
            dream_threshold=dream_threshold,
            custom_f_steps=scheduler_steps,
            custom_f=self.scheduler_step if enable_scheduler else lambda: None,
            param_f=CompareLatent.param_f_image,
            dreaming_batch_size=1,
            optimizer=self.get_dream_optim(),
            empty_dream_dataset=dream_sets.DreamDataset(transform=dream_transform),
            disable_transforms=disable_transforms
        )

        model.to('cuda:0').eval()
        constructed_dream = dream_module._generate_dreams_for_target(
            model, 
            used_class, 
            iterations=1, 
            update_progress_f=lambda *args, **kwargs: None, 
            task_index=0,
        )

        self._log(constructed_dreams=constructed_dream, logger=logger, label=label)

    def _log(self, constructed_dreams, logger, label):
        point_from_model = torch.from_numpy(self.point_from_model[-1]).cpu().squeeze()
        logged_main_point = self.logged_main_point[0].detach().cpu().squeeze()
        print('COMPARE LATENT: model point', point_from_model)
        print('COMPARE LATENT: main point', logged_main_point)

        diff = self.custom_loss_f(point_from_model, logged_main_point)
        point_from_model = point_from_model.numpy().squeeze()
        logged_main_point = logged_main_point.numpy().squeeze()
        table_model = wandb.Table(columns=[f'd{i}' for i in range(len(point_from_model))])
        print(logged_main_point.ndim)
        if(logged_main_point.ndim == 0):
            table_main = wandb.Table(columns=f'd{logged_main_point}')
        else:
            table_main = wandb.Table(columns=[f'd{i}' for i in range(len(logged_main_point))])
        table_model.add_data(*point_from_model)
        table_main.add_data(*logged_main_point)

        wandb.log({f'{label}/point_from_model': table_model})
        wandb.log({f'{label}/main_point': table_main})

        wandb.log({f'{label}/lossf_diff': diff})
        print('loss_f_diff', diff)
        logger.log_metrics({f'{label}/constructed_dreams': [
                wandb.Image(
                    i,
                ) for i in constructed_dreams
            ]
        })