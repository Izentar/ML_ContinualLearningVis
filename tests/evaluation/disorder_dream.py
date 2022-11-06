import torch
from dataset import dream_sets
from utils.functional import target_processing
from utils.functional.dream_objective import dream_objective_latent_lossf_creator
import wandb
from torch.optim.lr_scheduler import StepLR
from tests.evaluation.utils import CustomDreamDataModule
from utils.data_manipulation import select_class_indices_tensor
import random
import numpy as np

class DisorderDream():
    """
        Create image from the latent point created by model.
        It is done by sampling image from dataset by selected class, then image goes through the model giving latent point.
        After that do custom dreaming based on that point but use the previous selected image as the starting base for dreaming.
        You can compare sampled and generated image in the logs.
    """
    def __init__(self) -> None:
        self.logged_main_point = [0]

    def target_processing_custom_creator(point):
        def target_processing_custom(*args, **kwargs):
            return point
        return target_processing_custom

    def get_task_processing(self):
        return target_processing.target_processing_latent_decode

    def starting_image_creator(detached_image):
        real_image = torch.clone(detached_image).detach()
        real_img_shape = real_image.shape        
        real_image = torch.flatten(real_image).requires_grad_(True) 
        #to_pil = transforms.ToPILImage()
        def starting_image(**kwargs):
            def param_f():
                # uses 2D Fourier coefficients
                # sd - scale of the random numbers [0, 1)
                return [real_image], lambda: real_image.view(real_img_shape)
            return param_f
        return starting_image

    def wrapper_select_dream_tasks_f(used_class):
        def select_dream_tasks_f(tasks, task_index):
            return set(used_class)
        return select_dream_tasks_f

    def scheduler_step(self):
        self.scheduler.step()
        print('Scheduler step', self.scheduler._last_lr)

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

    def _select_rand_image(self, dataset, used_class, label):
        indices = select_class_indices_tensor(used_class, torch.IntTensor(dataset.targets))
        index = random.choice(indices)
        wandb.log({f'{label}/index_of_selected_img': index})
        return dataset[index]

    def _disorder_image(self, logger, label, image, device, disorder_input_image):
        logger.log_metrics({
            f'{label}/original_image': wandb.Image(image)
        })

        if(disorder_input_image):
            random = torch.rand(image.shape).to(device)
            image *= random
        return image

    def _compare_orig_constructed(self, original_im, constructed_im, label, logger):
        heatmap = torch.abs(original_im - constructed_im).squeeze()
        logger.log_metrics({
            f'{label}/heatmap_abs_compare': wandb.Image(torch.sum(heatmap, dim=0), mode='L')
        })
        logger.log_metrics({
            f'{label}/heatmap_squared_compare': wandb.Image(torch.sum(heatmap **2, dim=0), mode='L')
        })

    def __call__(
            self, 
            model, 
            used_class, 
            dataset, 
            logger, 
            dream_transform, 
            objective_f=None, 
            dream_loss_f=None, 
            disorder_input_image=True,
            enable_scheduler=True, # in experiments it does not changes output significantly but the loss converge to zero
            scheduler_steps=(1024*3, 1024*4, 1024*5),
            dream_threshold=(1024*6,),
        ):
        label = 'disorder_dream'
        point = None
        device = 'cuda:0'

        custom_loss_f = torch.nn.MSELoss() if dream_loss_f is None else dream_loss_f
        custom_objective_f = objective_f if objective_f is not None else dream_objective_latent_lossf_creator(
            logger=logger, 
            loss_f=custom_loss_f, 
            label=label,
        )

        # select image
        image, _ = self._select_rand_image(dataset, used_class, label=label)

        image = torch.unsqueeze(image, dim=0)
        image = image.to(device)
        model.to(device).eval()

        with torch.no_grad():
            model.eval()
            point = model(image)

        point.to(device)

        original_image = image.clone()

        image = self._disorder_image(
            logger=logger,
            label=label,
            image=image,
            device=device,
            disorder_input_image=disorder_input_image,
        )

        dream_module = CustomDreamDataModule(
            train_tasks_split=[[used_class]],
            select_dream_tasks_f=DisorderDream.wrapper_select_dream_tasks_f(used_class),
            dream_objective_f=custom_objective_f,
            target_processing_f=self.target_processing_decorator(DisorderDream.target_processing_custom_creator(point)),
            dreams_per_target=1,
            dream_threshold=dream_threshold,
            custom_f_steps=scheduler_steps,
            custom_f=self.scheduler_step if enable_scheduler else lambda: None,
            param_f=DisorderDream.starting_image_creator(detached_image=image),
            const_target_images_per_dreaming_batch=1,
            optimizer=self.get_dream_optim(),
            empty_dream_dataset=dream_sets.DreamDataset(transform=dream_transform),
            disable_transforms=True,
        )

        constructed_dream = dream_module._generate_dreams_for_target(
            model, 
            used_class, 
            iterations=1, 
            update_progress_f=lambda *args, **kwargs: None, 
            task_index=0,
        )
        self._compare_orig_constructed(
            original_im=original_image,
            constructed_im=constructed_dream.to(device),
            label=label,
            logger=logger,
        )

        self._log(constructed_dreams=constructed_dream, real_image=image, used_class=used_class, logger=logger, label=label)

    def _log(self, constructed_dreams, real_image, used_class, logger, label):
        logged_main_point = self.logged_main_point[0].detach().cpu().squeeze()
        print('main ', logged_main_point)
        logged_main_point = logged_main_point.numpy().squeeze()
        columns = [f'd{i}' for i in range(len(logged_main_point))]
        logged_main_point = np.append(logged_main_point, used_class)
        table_main = wandb.Table(columns=columns + ['class_type'])
        table_main.add_data(*logged_main_point)

        wandb.log({f'{label}/main_point': table_main})

        logger.log_metrics({f'{label}/constructed_dreams': [
                wandb.Image(
                    i,
                ) for i in constructed_dreams
            ]
        })

        logger.log_metrics({
            f'{label}/used_image': wandb.Image(real_image)
        })