import torch
from torchvision import transforms
from dataset import dream_sets
from dataset.CLModule import DreamDataModule
from utils.functional import task_processing
from utils.functional.dream_objective import SAE_island_dream_objective_f_creator
from lucent.optvis import param
import wandb
from lucent.optvis.objectives import wrap_objective, handle_batch

class CustomDreamDataModule(DreamDataModule):
    def prepare_data(self):
        raise Exception("Not implemented")

    def train_dataloader(self):
        raise Exception("Not implemented")

    def clear_dreams_dataset(self):
        raise Exception("Not implemented")

    def val_dataloader(self):
        raise Exception("Not implemented")

    def test_dataloader(self):
        raise Exception("Not implemented")

    def setup_tasks(self) -> None:
        raise Exception("Not implemented")

    def setup_task_index(self, task_index: int, loop_index: int) -> None:
        raise Exception("Not implemented")


def param_f_image(image_size, target_images_per_dreaming_batch, **kwargs):
    def param_f():
        # uses 2D Fourier coefficients
        # sd - scale of the random numbers [0, 1)
        return param.image(
            image_size, batch=target_images_per_dreaming_batch, sd=0.4
        )
    return param_f

def compare_latent(model, model_latent, used_class, logger, objective_f=None):
    logged_main_point = [0]
    constructed_dreams = []
    point_from_model = [0]

    def select_dream_tasks_f(tasks, task_index):
        return set(used_class)

    def task_processing_decorator(f):
        def wrapper(target, model, *args, **kwargs):
            point = f(target, model, *args, **kwargs)
            logged_main_point[0] = point
            return point
        return wrapper

    def custom_objective_f(target, target_point, model, **kwargs):
        @wrap_objective()
        def latent_objective_channel(target, target_layer, target_val, batch=None):
            loss_f = torch.nn.MSELoss() 
            @handle_batch(batch)
            def inner(model):
                latent = model(target_layer) # return feature map
                point_from_model[0] = latent
                latent_target = target_val.repeat(len(latent), 1)
                loss = loss_f(latent, latent_target)
                return loss
            return inner
        return latent_objective_channel(
            target_layer=model.get_objective_target(), 
            target_val=target_point.to(model.device),  
            target=target
        )
    
    tasks_processing_f = task_processing_decorator(task_processing.island_tasks_processing)
    objective_f = custom_objective_f

    dream_module = CustomDreamDataModule(
        train_tasks_split=[used_class],
        select_dream_tasks_f=select_dream_tasks_f,
        dream_objective_f=objective_f,
        tasks_processing_f=tasks_processing_f,
        dreams_per_target=1,
        dream_threshold=(1024*5, ),
        param_f=param_f_image,
        const_target_images_per_dreaming_batch=1,
        empty_dream_dataset=dream_sets.DreamDataset(transform=transforms.Compose([transforms.ToTensor()])),
    )

    model.to('cuda:0').eval()
    def empty_f():
        return
    constructed_dream = dream_module._generate_dreams_for_target(
        model, 
        used_class, 
        iterations=1, 
        update_progress_f=empty_f, 
        task_index=0
    )
    constructed_dreams.append(constructed_dream)

    point_from_model = point_from_model[0].detach().cpu()
    logged_main_point = logged_main_point[0].detach().cpu()

    diff = torch.abs(point_from_model[0].cpu() - logged_main_point[0].cpu()).mean()
    point_from_model = point_from_model.numpy().squeeze()
    logged_main_point = logged_main_point.numpy().squeeze()
    table_model = wandb.Table(columns=[f'd{i}' for i in range(len(point_from_model))])
    table_main = wandb.Table(columns=[f'd{i}' for i in range(len(logged_main_point))])
    table_model.add_data(*point_from_model)
    table_main.add_data(*logged_main_point)

    wandb.log({'compare_latent/point_from_model': table_model})
    wandb.log({'compare_latent/main_point': table_main})

    wandb.log({'compare_latent/abs_diff': diff})
    logger.log_metrics({'compare_latent/constructed_dreams': [
            wandb.Image(
                i,
            ) for i in constructed_dreams
        ]
    })