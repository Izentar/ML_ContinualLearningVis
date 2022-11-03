import torch
from torchvision import transforms
from dataset import dream_sets
from dataset.CLModule import DreamDataModule
from utils.functional import target_processing
from utils.functional.dream_objective import dream_objective_latent_lossf_creator
from lucent.optvis import param
import wandb
from lucent.optvis.objectives import wrap_objective, handle_batch
from torch.optim.lr_scheduler import StepLR
from tests.evaluation.utils import CustomDreamDataModule


class CompareLatent():
    """
        Compare latent points.
        Sample point in latent space, then do dreaming on this point.
        After that compare the point from generated image to the previous point.
        This is done by creating custom objective function that compares main point to the previous point 
        using provided loss function (default MSELoss).
    """
    def __init__(self):
        self.scheduler = None
        self.logged_main_point = [0]
        self.point_from_model = [0]
        self.custom_loss_f = torch.nn.MSELoss() 

    def param_f_image(image_size, target_images_per_dreaming_batch, **kwargs):
        def param_f():
            # uses 2D Fourier coefficients
            # sd - scale of the random numbers [0, 1)
            return param.image(
                image_size, batch=target_images_per_dreaming_batch, sd=0.4, #fft=False
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

    def custom_objective_f(self, target, target_point, model, **kwargs):
        @wrap_objective()
        def inner_obj_latent(target, target_layer, target_val, batch=None):
            loss_f = self.custom_loss_f
            @handle_batch(batch)
            def inner(model):
                latent = model(target_layer) # return feature map
                self.point_from_model[0] = latent
                latent_target = target_val.repeat(len(latent), 1)
                #print(latent[0][0], latent_target[0][0])
                loss = loss_f(latent, latent_target)
                return loss

            def inner_choose_random_sample(model_layers):
                latent = model_layers(target_layer) # return feature map
                out_target_val = target_processing.target_processing_latent_sample_normal_std(target_val, torch.ones_like(target_val) * 0.2)
                self.point_from_model[0] = latent
                latent_target = out_target_val.repeat(len(latent), 1).to(latent.device)
                #print(latent[0][0], latent_target[0][0])
                loss = loss_f(latent, latent_target)
                return loss
            return inner_choose_random_sample
        return inner_obj_latent(
            target_layer=model.get_objective_target(), 
            target_val=target_point.to(model.device),  
            target=target
        )

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

    def __call__(self, model, used_class, logger, dream_transform, target_processing_f, loss_f=None):
        constructed_dreams = []

        custom_target_processing_f = self.target_processing_decorator(target_processing_f)

        self.custom_loss_f = self.custom_loss_f if loss_f is None else loss_f
        objective_fun = self.custom_objective_f

        dream_module = CustomDreamDataModule(
            train_tasks_split=[used_class],
            select_dream_tasks_f=CompareLatent.wrapper_select_dream_tasks_f(used_class),
            dream_objective_f=objective_fun,
            target_processing_f=custom_target_processing_f,
            dreams_per_target=1,
            dream_threshold=(1024*6,),
            custom_f_steps=(1024*3,1024*5),
            custom_f=self.scheduler_step,
            param_f=CompareLatent.param_f_image,
            const_target_images_per_dreaming_batch=1,
            optimizer=self.get_dream_optim(),
            empty_dream_dataset=dream_sets.DreamDataset(transform=dream_transform),
        )

        model.to('cuda:0').eval()
        constructed_dream = dream_module._generate_dreams_for_target(
            model, 
            used_class, 
            iterations=1, 
            update_progress_f=lambda *args, **kwargs: None, 
            task_index=0,
        )
        constructed_dreams.append(constructed_dream)

        self._log(constructed_dreams=constructed_dreams, logger=logger)

    def _log(self, constructed_dreams, logger):
        point_from_model = self.point_from_model[0].detach().cpu().squeeze()
        logged_main_point = self.logged_main_point[0].detach().cpu().squeeze()
        print('model', point_from_model)
        print('main ', logged_main_point)

        diff = self.custom_loss_f(point_from_model, logged_main_point)
        point_from_model = point_from_model.numpy().squeeze()
        logged_main_point = logged_main_point.numpy().squeeze()
        table_model = wandb.Table(columns=[f'd{i}' for i in range(len(point_from_model))])
        table_main = wandb.Table(columns=[f'd{i}' for i in range(len(logged_main_point))])
        table_model.add_data(*point_from_model)
        table_main.add_data(*logged_main_point)

        label = 'compare_latent'
        wandb.log({f'{label}/point_from_model': table_model})
        wandb.log({f'{label}/main_point': table_main})

        wandb.log({f'{label}/loss_f_diff': diff})
        print('loss_f_diff', diff)
        logger.log_metrics({f'{label}/constructed_dreams': [
                wandb.Image(
                    i,
                ) for i in constructed_dreams
            ]
        })