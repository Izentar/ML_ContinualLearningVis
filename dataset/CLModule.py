import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
from pytorch_lightning import LightningDataModule, LightningModule
from utils import data_manipulation as datMan
from dataset import dream_sets
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from lucent.optvis import param, render

import numpy as np
from math import ceil
from abc import ABC, abstractmethod


class BaseCLDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_tasks(self) -> None:
        pass

    @abstractmethod
    def setup_task_index(self, task_index: int) -> None:
        pass

    def generate_synthetic_data(self, model: LightningModule, task_index: int) -> None:
        pass

class CLDataModule(BaseCLDataModule):
    def __init__(
        self,
        train_tasks_split,
        dreams_per_target,
        dataset_class,
        select_dream_tasks_f,
        dream_objective_f,
        tasks_processing_f=datMan.default_tasks_processing,
        images_per_dreaming_batch=8,
        val_tasks_split=None,
        dream_transforms=None,
        max_logged_dreams=8, 
        fast_dev_run=False,
        batch_size=32,
        num_workers=4,
        shuffle=True,
        dream_num_workers=None,
        test_val_num_workers=None,
        dream_shuffle=None,
        root="data",
    ):
        """
        Args:
            task_split: list containing list of class indices per task
            select_dream_tasks_f: function f(tasks:list, task_index:int) that will be used to select 
                the tasks to use during dreaming. If function uses more parameters, use lambda expression. 
            dream_objective_f: function f(target: list, model: model.base.CLBase, <source CLDataModule object, self>)
            tasks_processing_f: function that will take the list of targets and process them to the desired form. 
                The default function do nothing to targets. 
        """
        super().__init__()
        self.train_tasks_split = train_tasks_split
        self.tasks_processing_f = tasks_processing_f
        self.val_tasks_split = (val_tasks_split if val_tasks_split is not None else train_tasks_split)
        self.dreams_per_target = dreams_per_target
        self.images_per_dreaming_batch = images_per_dreaming_batch
        self.fast_dev_run = fast_dev_run
        self.select_dream_tasks_f = select_dream_tasks_f
        self.dream_objective_f = dream_objective_f

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle

        self.dream_num_workers = dream_num_workers if dream_num_workers is not None else num_workers
        self.dream_shuffle = dream_shuffle if dream_shuffle is not None else shuffle
        self.test_val_num_workers = test_val_num_workers if test_val_num_workers is not None else num_workers
        self.root = root

        self.train_task = None
        self.dream_task = None
        # TODO
        self.image_size = 32
        self.max_logged_dreams = max_logged_dreams
        self.dreams_dataset = dream_sets.DreamDataset(transform=dream_transforms)
        self.dataset_class = dataset_class

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = self.dataset_class(
            root=self.root, train=True, transform=transform, download=True
        )
        test_dataset = self.dataset_class(root=self.root, train=False, transform=transform)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.setup_tasks()

    def setup_tasks(self):
        self.train_datasets = self._split_dataset(
            self.train_dataset, self.train_tasks_split
        )
        self.test_datasets = self._split_dataset(
            self.test_dataset, self.val_tasks_split
        )

    @staticmethod
    def _split_dataset(dataset, tasks_split):
        split_dataset = []
        for current_classes in tasks_split:
            task_indices = np.isin(np.array(dataset.targets), current_classes)
            split_dataset.append(Subset(dataset, np.where(task_indices)[0]))
        return split_dataset

    def setup_task_index(self, task_index: int) -> None:
        """
            Choose the index of the task that will be used during training.
        """
        #TODO - to log
        print(f"Selected task number: {task_index}")
        self.train_task = self.train_datasets[task_index]
        self.dream_task = self.dreams_dataset if len(self.dreams_dataset) > 0 else None

    def train_dataloader(self):
        """
            Returns the dictionary of :
            - "normal": normal_loader
            - "dream": dream_loader [Optional]
        """
        # check
        if self.train_task is None:
            raise Exception("No task index set for training")
        dream_loader = None
        if self.dream_task:
            # first loop is always False. After accumulating dreams this dataloader will be used.
            dream_loader = DataLoader(
                self.dream_task, batch_size=self.batch_size, num_workers=self.dream_num_workers, shuffle=self.dream_shuffle
            )
        normal_loader = DataLoader(
            self.train_task,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=self.shuffle,
        )
        if dream_loader is not None:
            return {"normal": normal_loader, "dream": dream_loader}
        else:
            return {"normal": normal_loader}

    def clear_dreams_dataset():
        self.dreams_dataset.clear()

    def val_dataloader(self):
        return [
            DataLoader(dataset, batch_size=self.batch_size, num_workers=self.test_val_num_workers)
            for dataset in self.test_datasets
        ]

    def test_dataloader(self):
        return DataLoader(
            ConcatDataset(self.train_datasets), batch_size=self.batch_size, num_workers=self.test_val_num_workers
        )

    def generate_synthetic_data(self, model: LightningModule, task_index):
        """Generate new dreams."""
        dream_targets = self.select_dream_tasks_f(self.train_tasks_split, task_index)
        dream_targets = self.tasks_processing_f(dream_targets)

        model_mode = model.training # from torch.nn.Module
        if model_mode:
            model.eval()

        iterations = ceil(self.dreams_per_target / self.images_per_dreaming_batch)
        new_dreams, new_targets = self.__generate_dreams(
            model=model,
            dream_targets=dream_targets, 
            iterations=iterations
        )

        self.__log_fast_dev_run(new_dreams=new_dreams, new_targets=new_targets)

        self.dreams_dataset.extend(new_dreams, new_targets)

        if model_mode:
            model.train()

    def __generate_dreams(self, model, dream_targets, iterations):
        new_dreams = []
        new_targets = []
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(complete_style="magenta"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            dreaming_progress = progress.add_task(
                "[bright_blue]Dreaming...", total=(len(dream_targets) * iterations)
            )

            for target in dream_targets: #TODO do we really need sorted targets? It throws error for point like target
                target_progress = progress.add_task(
                    f"[bright_red]Class: {target}\n", total=iterations
                )

                def update_progress():
                    progress.update(target_progress, advance=1)
                    progress.update(dreaming_progress, advance=1)

                target_dreams = self.__generate_dreams_for_target(
                    model=model, 
                    target=target, 
                    iterations=iterations, 
                    update_progress=update_progress
                )
                new_targets.extend([target] * target_dreams.shape[0])
                new_dreams.append(target_dreams)
                progress.remove_task(target_progress)
        new_dreams = torch.cat(new_dreams)
        new_targets = torch.tensor(new_targets)
        return new_dreams, new_targets

    def __log_fast_dev_run(self, new_dreams, new_targets):
        if not self.fast_dev_run:
            num_dreams = new_dreams.shape[0]
            wandb.log(
                {
                    "examples": [
                        wandb.Image(
                            new_dreams[i, :, :, :],
                            caption=f"sample: {i} target: {new_targets[i]}",
                        )
                        for i in range(min(num_dreams, self.max_logged_dreams))
                    ]
                }
            )

    def __generate_dreams_for_target(self, model, target, iterations, update_progress):
        dreams = []
        for _ in range(iterations):

            def batch_param_f():
                # uses 2D Fourier coefficients
                # sd - scale of the random numbers [0, 1)
                return param.image(
                    self.image_size, batch=self.images_per_dreaming_batch, sd=0.4
                )
            objective = self.dream_objective_f(target, model, self)

            numpy_render = torch.from_numpy(
                render.render_vis(
                    model=model,
                    objective_f=objective,
                    param_f=batch_param_f,
                    fixed_image_size=self.image_size,
                    progress=False,
                    show_image=False,
                    #TODO - maybe use other params
                    #transforms=None, # default use standard transforms on the image that is generating.
                    #optimizer=None, # default Adam
                    #preprocess=True, # simple ImageNet normalization that will be used on the image that is generating.
                    #thresholds=(512,) # max() - how many iterations is used to generate an input image. Also
                                        # for the rest of the numbers, save the image during the i-th iteration
                )[-1] # return the last, most processed image (thresholds)
            )

            dreams.append(torch.permute(numpy_render, (0, 3, 1, 2)))
            update_progress()
        return torch.cat(dreams)



