import torch
from datamodule import dream_sets
from utils.functional import target_processing
from utils.functional.dream_objective import dream_objective_latent_lossf_creator
import wandb
from torch.optim.lr_scheduler import StepLR
from tests.evaluation.utils import CustomDreamDataModule
from utils.data_manipulation import select_class_indices_tensor
import random
import numpy as np
from utils.counter import CounterBase, Counter
from torchvision import transforms
from dream.image import Image
from torchvision import transforms as tr

class DisorderDream():
    """
        Create image from the latent point created by model.
        It is done by sampling image from datamodule by selected class, then image goes through the model giving latent point.
        After that do custom dreaming based on that point but use the previous selected image as the starting base for dreaming.
        You can compare sampled and generated image in the logs.
    """
    def __init__(self) -> None:
        self.logged_main_point = [0]
        self.class_label = 'disorder_dream'

    def target_processing_custom_creator(point):
        def target_processing_custom(*args, **kwargs):
            return point
        return target_processing_custom

    def get_task_processing(self):
        return target_processing.target_processing_latent_decode

    def starting_image_creator(image):
        def reinit():
            tmp = torch.clone(image).detach().requires_grad_(True)
            def image_f():
                return tmp
            return image_f, [tmp]
        
        def starting_image(**kwargs):
            return Image(dtype='custom', reinit_f=reinit)
        return starting_image

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

    def _select_rand_image(self, dataset, used_class, label):
        indices = select_class_indices_tensor(used_class, torch.IntTensor(dataset.targets))
        index = random.choice(indices)
        wandb.log({f'{label}/index_of_selected_img': index})
        return dataset[index]

    def _disorder_image(self, image, device, disorder_input_image, sigma_disorder, start_img_value=None):
        image = image.clone()
        if(disorder_input_image):
            mean = torch.zeros_like(image)
            if(sigma_disorder == 0.):
                print(f"Disorder dreams: sigma value is equal zero: {sigma_disorder}")
                random = mean
            else:
                std = torch.ones_like(image) * sigma_disorder
                random = torch.normal(mean, std).to(device)
            if(start_img_value is not None):
                print(f"Disorder dreams: created uniform image with value: {start_img_value}. Original image was deleted.")
                image = torch.ones_like(image) * start_img_value
            image += random
            
        return image

    def _compare_orig_constructed(self, original_img, disorder_img, constructed_img, label, logger):
        assert original_img.shape == constructed_img.shape, f"Bad shape: orig {original_img.shape} constr {constructed_img.shape}"
        assert disorder_img.shape == constructed_img.shape, f"Bad shape: disorder {disorder_img.shape} constr {constructed_img.shape}"
        heatmap = torch.abs(original_img - constructed_img).squeeze()
        to_pil = lambda img: transforms.ToPILImage()(img)
        
        logger.log_metrics({
            f'{label}/heatmap_abs_compare(original - constructed)': wandb.Image(to_pil(torch.sum(heatmap, dim=0)), mode='L')
        })
        logger.log_metrics({
            f'{label}/heatmap_squared_compare(original - constructed)': wandb.Image(to_pil(torch.sum(heatmap **2, dim=0)), mode='L')
        })
        
        heatmap = torch.abs(disorder_img - constructed_img).squeeze()
        logger.log_metrics({
            f'{label}/heatmap_abs_compare(disorder - constructed)': wandb.Image(to_pil(torch.sum(heatmap, dim=0)), mode='L')
        })
        logger.log_metrics({
            f'{label}/heatmap_squared_compare(disorder - constructed)': wandb.Image(to_pil(torch.sum(heatmap **2, dim=0)), mode='L')
        })

    def __call__(
            self, 
            model, 
            used_class, 
            dataset, 
            logger, 
            dream_transform, 
            sigma_disorder,
            objective_f=None, 
            dream_loss_f=None, 
            disorder_input_image=True,
            scheduler_milestones=None, # in experiments it does not changes output significantly but the loss converge to zero
            dream_threshold=(1024*2,),
            start_img_value=None,
            device = 'cuda:0'
        ):
        point = None

        custom_loss_f = torch.nn.MSELoss() if dream_loss_f is None else dream_loss_f
        custom_objective_f = objective_f if objective_f is not None else dream_objective_latent_lossf_creator(
            logger=logger, 
            loss_f=custom_loss_f, 
            label=self.class_label,
        )

        # select image
        image, _ = self._select_rand_image(dataset, used_class, label=self.class_label)

        image = torch.unsqueeze(image, dim=0)
        image = image.to(device)
        model.to(device).eval()

        with torch.no_grad():
            model.eval()
            point = model(image)

        point.to(device)
        print('Latent point generated from model', point)

        original_image = image.clone()

        disorder_image = self._disorder_image(
            image=image,
            device=device,
            disorder_input_image=disorder_input_image,
            sigma_disorder=sigma_disorder,
            start_img_value=start_img_value,
        )

        dream_image_f = DisorderDream.starting_image_creator(image=disorder_image)

        cfg_map={
            'cfg_vis': CustomDreamDataModule.Visualization(
                per_target=1,
                threshold=dream_threshold,
                batch_size=1,
                disable_transforms=True,
            ),
            'cfg': CustomDreamDataModule.Config(
                train_tasks_split=[[used_class]],
            ),
            'cfg_vis_optim': CustomDreamDataModule.Visualization.Optimizer(
                type='adam',
                kwargs={
                    'lr':5e-3,
                    'betas': (0.0, 0.0)
                },
            ),
            
        }

        if(scheduler_milestones is not None):
            cfg_map.update({
                'cfg_vis_sched': CustomDreamDataModule.Visualization.Scheduler(
                    type='MULTISTEP_SCHED',
                    kwargs={
                        'milestones': scheduler_milestones,
                        'gamma': 0.1,
                    },
                ),
            })

        dream_module = CustomDreamDataModule(
            cfg_map=cfg_map,
            select_dream_tasks_f=DisorderDream.wrapper_select_dream_tasks_f(used_class),
            dream_objective_f=custom_objective_f,
            target_processing_f=self.target_processing_decorator(DisorderDream.target_processing_custom_creator(point)),
            dream_image_f=dream_image_f,
            #param_f=DisorderDream.starting_image_creator(detached_image=image),
            empty_dream_dataset=dream_sets.DreamDataset(transform=dream_transform),
            logger=logger,
        )

        task_index = 0
        model.to(device).eval() # must be before rendervis_state or rendervis_state device can be on cpu
        custom_loss_gather_f, name = dream_module.get_custom_loss_gather_f(task_index=task_index)
        rendervis_state = dream_module.get_rendervis(model=model, custom_loss_gather_f=custom_loss_gather_f)
        rendervis_state.transform_f = tr.Lambda(lambda x: x.to(device))

        name[0] = f"{name[1]}/disorder_dream_target_{used_class}"

        constructed_dream = dream_module.generate_dreams_for_target(
            model, 
            target=used_class, 
            iterations=1, 
            rendervis_state=rendervis_state,
        )[0] # get first from tuple and left batch dim 

        self._compare_orig_constructed(
            original_img=original_image,
            disorder_img=disorder_image,
            constructed_img=constructed_dream.to(device),
            label=self.class_label,
            logger=logger,
        )

        self._log(constructed_dreams=constructed_dream, original_image=original_image, 
                  disorder_image=disorder_image, used_class=used_class, logger=logger, label=self.class_label)

    def _log(self, constructed_dreams, original_image, disorder_image, used_class, logger, label):
        logged_main_point = self.logged_main_point[0].detach().cpu().squeeze()
        print('processed main point:', logged_main_point)
        logged_main_point = logged_main_point.numpy().squeeze()
        columns = [f'd{i}' for i in range(len(logged_main_point))]
        logged_main_point = np.append(logged_main_point, used_class)
        table_main = wandb.Table(columns=columns + ['class_type'])
        table_main.add_data(*logged_main_point)

        wandb.log({f'{label}/main_point': table_main})

        logger.log_metrics({
            f'{label}/original_image': wandb.Image(original_image)
        })

        logger.log_metrics({f'{label}/constructed_dreams': [
                wandb.Image(
                    i,
                ) for i in constructed_dreams
            ]
        })

        logger.log_metrics({
            f'{label}/disorder_image': wandb.Image(disorder_image)
        })