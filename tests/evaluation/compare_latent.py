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
from dream.image import Image
from utils.utils import parse_image_size
from dream.custom_render import RenderVisState
from torchvision import transforms as tr


class CompareLatent():
    """
        Compare latent points.
        Sample point A in latent space, then do visualization using this point A.
        This visualization generates the second point B. Compare the point B from generated image to the previous sampled point A.
        This is done by creating custom objective function that compares main point to the sampled point 
        using provided loss function (default MSELoss).
    """
    def __init__(self):
        self.scheduler = None
        self.logged_main_point = [0]
        self.point_from_model = []
        self.custom_loss_f = torch.nn.MSELoss() 

    def param_f_image(self, dtype, image_size, dreaming_batch_size, decorrelate, **kwargs):
        channels, w, h = parse_image_size(image_size)
        return Image(dtype=dtype, w=w, h=h, channels=channels, batch=dreaming_batch_size,
            decorrelate=decorrelate)

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

    def __call__(
            self, 
            model, 
            used_class, 
            logger, 
            dream_fetch_transform, 
            target_processing_f=None, 
            loss_f=None,
            enable_scheduler=True, 
            scheduler_step_size=2048,
            dream_threshold=(1024*6,),
            loss_obj_step_sample=False,
            enable_transforms=True,
            device='cuda:0',
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
            )

        dream_image_f = self.param_f_image

        cfg_map={
            'cfg_vis': CustomDreamDataModule.Visualization(
                per_target=1,
                threshold=dream_threshold,
                batch_size=1,
                disable_transforms=not enable_transforms,
                image_type='pixel',
                standard_image_size=32,
            ),
            'cfg': CustomDreamDataModule.Config(
                train_tasks_split=[[used_class]],
            ),
            'cfg_vis_optim': CustomDreamDataModule.Visualization.Optimizer(
                kwargs={
                    'lr':5e-3
                },
            ),
            
        }

        if(enable_scheduler):
            cfg_map.update({
                'cfg_vis_sched': CustomDreamDataModule.Visualization.Scheduler(
                    type='STEP_SCHED',
                    kwargs={
                        'step_size': scheduler_step_size,
                        'gamma': 0.1,
                    },
                ),
            })

        dream_module = CustomDreamDataModule(
            cfg_map=cfg_map,
            select_dream_tasks_f=CompareLatent.wrapper_select_dream_tasks_f(used_class),
            dream_objective_f=objective_fun,
            target_processing_f=custom_target_processing_f,
            dream_image_f=dream_image_f,
            empty_dream_dataset=dream_sets.DreamDataset(transform=dream_fetch_transform),
            logger=logger,
        )

        task_index = 0

        model.to(device).eval() # must be before rendervis_state or rendervis_state device can be on cpu
        custom_loss_gather_f, name = dream_module.get_custom_loss_gather_f(task_index=task_index)
        rendervis_state = dream_module.get_rendervis(model=model, custom_loss_gather_f=custom_loss_gather_f)
        rendervis_state.transform_f = tr.Lambda(lambda x: x.to(device))

        name[0] = f"{name[1]}/compare_latent_target_{used_class}"

        start_image = dream_module.dream_image.image().clone()

        constructed_dream = dream_module.generate_dreams_for_target(
            model, 
            target=used_class, 
            iterations=1, 
            rendervis_state=rendervis_state,
        )[0]

        self._log(start_image=start_image, constructed_dreams=constructed_dream, logger=logger, label=label)

    def _log(self, start_image, constructed_dreams, logger, label):
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

        logger.log_metrics({f'{label}/start_images': [
                wandb.Image(
                    i,
                ) for i in start_image
            ]
        })