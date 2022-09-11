import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
from pytorch_lightning import LightningDataModule, LightningModule
from utils import data_manipulation as datMan
from utils.functional import task_processing
from dataset import dream_sets
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from lucent.optvis import param, render
import wandb

import numpy as np
from math import ceil
from abc import ABC, abstractmethod

from utils.data_manipulation import get_target_from_dataset
from model import base


class BaseCLDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_tasks(self) -> None:
        """
            Create a list of train and test datasets, where the indices 
            indicate the task that will be processed in the future.
        """
        pass

    @abstractmethod
    def setup_task_index(self, task_index: int) -> None:
        """
            Set the datasets that will be currently processed.
        """
        pass

    def generate_synthetic_data(self, model: LightningModule, task_index: int) -> None:
        """
            Generate new dreams from the tasks.
        """
        pass

class DreamDataModule(BaseCLDataModule, ABC):
    def __init__(self,
        train_tasks_split,
        select_dream_tasks_f,
        dream_objective_f,
        dreams_per_target,
        dream_threshold,
        tasks_processing_f=task_processing.default_tasks_processing,
        const_target_images_per_dreaming_batch=8,
        max_logged_dreams=8, 
        fast_dev_run=False,
        fast_dev_run_dream_threshold=None,
        image_size = 32,
        progress_bar = None,
        empty_dream_dataset=None,
    ):
        """
        Args:
            task_split: list containing list of class indices per task
            select_dream_tasks_f: function f(tasks:list, task_index:int) that will be used to select 
                the tasks to use during dreaming. If function uses more parameters, use lambda expression as adapter. 
            dream_objective_f: function f(target: list, model: model.base.CLBase, <source CLDataModule object, self>)
            tasks_processing_f: function that will take the list of targets and process them to the desired form.
                For example it will take a task and transform it to the point from normal distribution.
                The default function do nothing to targets. 
        """
        super().__init__()
        self.train_tasks_split = train_tasks_split
        self.select_dream_tasks_f = select_dream_tasks_f
        self.tasks_processing_f = tasks_processing_f
        self.dreams_per_target = dreams_per_target
        self.const_target_images_per_dreaming_batch = const_target_images_per_dreaming_batch
        self.image_size = image_size
        self.dream_objective_f = dream_objective_f
        self.max_logged_dreams = max_logged_dreams
        self.progress_bar = progress_bar

        self.dreams_dataset = empty_dream_dataset
        self.calculated_mean_std = False
        self.tasks_ids = None

        self.fast_dev_run_dream_threshold = fast_dev_run_dream_threshold if isinstance(fast_dev_run_dream_threshold, tuple) else (fast_dev_run_dream_threshold, )
        self.dream_threshold = dream_threshold if isinstance(dream_threshold, tuple) else (dream_threshold, )
        self.fast_dev_run = fast_dev_run
        
        if(empty_dream_dataset is None):
            raise Exception("Empty dream dataset.")
    

    @abstractmethod
    def prepare_data(self):
        pass

    @abstractmethod
    def train_dataloader(self):
        pass

    @abstractmethod
    def clear_dreams_dataset(self):
        pass

    @abstractmethod
    def val_dataloader(self):
        pass

    @abstractmethod
    def test_dataloader(self):
        pass

    def generate_synthetic_data(self, model: LightningModule, task_index: int) -> None:
        """Generate new dreams."""
        primal_dream_targets = self.select_dream_tasks_f(self.train_tasks_split, task_index)
        #dream_targets = self.transform_targets(model=model, dream_targets=primal_dream_targets, task_index=task_index)
        dream_targets = primal_dream_targets

        model_mode = model.training # from torch.nn.Module
        if model_mode:
            model.eval()

        iterations = ceil(self.dreams_per_target / self.const_target_images_per_dreaming_batch)
        new_dreams, new_targets = self.__generate_dreams(
            model=model,
            dream_targets=dream_targets, 
            iterations=iterations,
            task_index=task_index,
        )

        self.__log_fast_dev_run(new_dreams=new_dreams, new_targets=new_targets)

        self.dreams_dataset.extend(new_dreams, new_targets, model)

        self.calculated_mean_std = False
        if model_mode:
            model.train()

    def __generate_dreams_impl(self, model, dream_targets, iterations, task_index, new_dreams, new_targets, progress):
        dreaming_progress = progress.add_task(
            "[bright_blue]Dreaming...", total=(len(dream_targets) * iterations)
        )
        #print('dreaming_progress', dreaming_progress)
        tasks_ids = []
        tasks_ids.append(dreaming_progress)

        tasks_ids = []
        for target in sorted(dream_targets):
            target_progress = progress.add_task(
                f"[bright_red]Class: {target}\n", total=iterations
            )
            tasks_ids.append(target_progress)
            #print('target_progress', target_progress)

            def update_progress():
                progress.update(target_progress, advance=1)
                progress.update(dreaming_progress, advance=1)

            target_dreams = self.__generate_dreams_for_target(
                model=model, 
                target=target, 
                iterations=iterations, 
                update_progress=update_progress,
                task_index=task_index
            )
            new_targets.extend([target] * target_dreams.shape[0])
            new_dreams.append(target_dreams)
            #print('aaa', progress._tasks, '\n\n')
            #progress.remove_task(target_progress)
            #progress.stop_task(target_progress)
        #progress.remove_task(dreaming_progress)
        #print('bbb', progress._tasks, '\n\n')
        self.tasks_ids = tasks_ids
        
    def __generate_dreams(self, model, dream_targets, iterations, task_index):
        new_dreams = []
        new_targets = []

        if (self.progress_bar is not None):
            #self.progress_bar._stop_progress()
            self.__generate_dreams_impl(
                model=model, 
                dream_targets=dream_targets, 
                iterations=iterations, 
                task_index=task_index,
                new_dreams=new_dreams, 
                new_targets=new_targets, 
                progress=self.progress_bar.progress,
            )
            #self.progress_bar._init_progress()
        else:
            with Progress(
                "[progress.description]{task.description}",
                BarColumn(complete_style="magenta"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                self.__generate_dreams_impl(
                    model=model, 
                    dream_targets=dream_targets, 
                    iterations=iterations, 
                    task_index=task_index,
                    new_dreams=new_dreams, 
                    new_targets=new_targets, 
                    progress=progress,
                )
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

    def __generate_dreams_for_target(self, model, target, iterations, update_progress, task_index):
        dreams = []
        thresholds = self.fast_dev_run_dream_threshold if self.fast_dev_run and self.fast_dev_run_dream_threshold is not None else self.dream_threshold
        for _ in range(iterations):

            def batch_param_f():
                # uses 2D Fourier coefficients
                # sd - scale of the random numbers [0, 1)
                return param.image(
                    self.image_size, batch=self.const_target_images_per_dreaming_batch, sd=0.4
                )

            target_point = self.transform_targets(model=model, dream_target=target, task_index=task_index)
            objective = self.dream_objective_f(target=target, target_point=target_point, model=model, source_dataset_obj=self)

            # python3.10/site-packages/lucent/optvis/param/spatial.py
            # Here we use 'cuda:0' on the tensors. In multigpu env this may be troublesome.
            # The image is saved into spectrum_real_imag_t which changes after optim.step()
            # Optimizer will update only the input layer because of the only specified layer 'param' passed to it.
            # By default it will do 512 iterations
            # In objective_f we specify the layers which will be used to compute loss. In ChiLoss 
            #   we only need the last layer loss plus minor diversity on some convolutional layers.
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
                    thresholds=thresholds # max() - how many iterations is used to generate an input image. Also
                                        # for the rest of the numbers, save the image during the i-th iteration
                )[-1] # return the last, most processed image (thresholds)
            )

            dreams.append(torch.permute(numpy_render, (0, 3, 1, 2)))
            update_progress()
        return torch.cat(dreams)

    def transform_targets(self, model, dream_target, task_index) -> torch.Tensor:
        """
            Invoked after every new dreaming that produces an image.
            Override this if you need more than what the function tasks_processing_f itself can offer.
            Returns tensor representing target. It can be one number or an array of numbers (point)
        """
        return self.tasks_processing_f(target=dream_target, model=model)

    def clear_task_progress(self):
        # DO NOT USE, it throws error for some reason for RichProgressBar I dont know
        if(self.progress_bar is not None and self.tasks_ids is not None):
            for t in self.tasks_ids:
                self.progress_bar.progress.remove_task(t)
            self.tasks_ids = None

    def get_task_classes(self, task_number):
        return self.train_tasks_split[task_number]

class CLDataModule(DreamDataModule):
    def __init__(
        self,
        dataset_class,
        val_tasks_split=None,
        max_logged_dreams=8, 
        batch_size=32,
        dream_batch_size=8,
        num_workers=4,
        shuffle=True,
        steps_to_locate_mean = None,
        dream_num_workers=None,
        test_val_num_workers=None,
        dream_shuffle=None,
        datasampler=None,
        root="data",
        **kwargs
    ):
        """
            dataset_class: pytorch dataset like CIFAR10. Pass only a class, not an object.
            steps_to_locate_mean: how many iterations you want to do search for mean and std in current train dataset.
                It may be an intiger or float from [0, 1].
            datasampler: pytorch batch sampler. Pass the Class or lambda adapter of the signature 
                f(dataset, batch_size, shuffle, classes), where classes is a unique set of 
                all classes that are present in dataset. Example (pass this as datasampler)
                def f(dataset, batch_size, shuffle, classes): 
                    return PairingBatchSampler(
                        dataset=dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        classes=classes,
                        main_class_split=0.55,
                        classes_frequency=[1 / len(classes)] * len(classes)
                    )
        """
        super().__init__(**kwargs)
        self.val_tasks_split = (val_tasks_split if val_tasks_split is not None else self.train_tasks_split)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.steps_to_locate_mean = steps_to_locate_mean

        self.dream_num_workers = dream_num_workers if dream_num_workers is not None else num_workers
        self.dream_shuffle = dream_shuffle if dream_shuffle is not None else shuffle
        self.test_val_num_workers = test_val_num_workers if test_val_num_workers is not None else num_workers
        self.datasampler = datasampler if datasampler is not None else None
        self.root = root
        self.dream_batch_size = dream_batch_size

        self.train_task = None
        self.dream_dataset_current_task = None
        self.current_task_index = None
        # TODO
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

    def _get_subset(dataset, classes: list):
        task_indices = np.isin(np.array(dataset.targets), classes)
        return Subset(dataset, np.where(task_indices)[0])

    def __setup_dream_dataset(self):
        if self.dreams_dataset is not None and not self.dreams_dataset.empty():
            self.dream_dataset_current_task = self.dreams_dataset 
        else: 
            self.dream_dataset_current_task = None

    def setup_task_index(self, task_index: int) -> None:
        """
            Choose the index of the task that will be used during training.
        """
        #TODO - to log
        if task_index == self.current_task_index: # guard
            return
        if task_index >= len(self.train_datasets):
            raise Exception(f"Index {task_index} out of range {len(self.train_datasets)}")
        print(f"Selected task number: {task_index}")
        self.current_task_index = task_index
        self.train_task = self.train_datasets[task_index]
        self.__setup_dream_dataset()

        if(self.train_task is None or len(self.train_task) <= 0):
            raise Exception(f"Train task dataset not set properly. Used index {task_index} from {len(self.train_datasets)}")

    def generate_synthetic_data(self, model: LightningModule, task_index: int) -> None:
        super().generate_synthetic_data(model=model, task_index=task_index)
        self.__setup_dream_dataset()

    def train_dataloader(self):
        """
            Returns the dictionary of :
            - "normal": normal_loader
            - "dream": dream_loader [Optional position]
        """
        if self.train_task is None: # check
            raise Exception("No task index set for training")
        dream_loader = None
        if self.dream_dataset_current_task:
            # first loop is always False. After accumulating dreams this dataloader will be used.
            shuffle = self.dream_shuffle if self.datasampler is None else False
            # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last

            #dream_tasks_classes = self.val_tasks_split[self.current_task_index]
            if(self.current_task_index < 1):
                raise Exception("Cannot create dream dataloader. No tasks were done before / no data avaliable.")
            dream_tasks_classes = set()
            
            for i in range(self.current_task_index):
                dream_tasks_classes = dream_tasks_classes.union(self.val_tasks_split[i])
            dream_tasks_classes = list(dream_tasks_classes)

            dream_loader = DataLoader(
                self.dream_dataset_current_task, 
                batch_size=self.batch_size if self.datasampler is None else 1, 
                num_workers=self.dream_num_workers, 
                shuffle=shuffle if self.datasampler is None else False, 
                pin_memory=True,
                batch_sampler=self.datasampler(
                    dataset=self.dream_dataset_current_task, 
                    shuffle=self.dream_shuffle,
                    classes=dream_tasks_classes,
                    batch_size=self.dream_batch_size,
                ) if self.datasampler is not None else None
            )

        shuffle = self.shuffle if self.datasampler is None else False
        normal_loader = DataLoader(
            self.train_task,
            batch_size=self.batch_size if self.datasampler is None else 1,
            num_workers=self.num_workers,
            shuffle=shuffle if self.datasampler is None else False, 
            pin_memory=True,
            batch_sampler=self.datasampler(
                    dataset=self.train_task, 
                    shuffle=self.shuffle,
                    classes=self.val_tasks_split[self.current_task_index],
                    batch_size=self.batch_size,
                ) if self.datasampler is not None else None
        )
        if dream_loader is not None:
            return {"normal": normal_loader, "dream": dream_loader}
        else:
            return {"normal": normal_loader}

    def clear_dreams_dataset(self):
        self.dreams_dataset.clear()

    def val_dataloader(self):
        return [
            DataLoader(dataset, 
            batch_size=self.batch_size if self.datasampler is None else 1, 
            num_workers=self.test_val_num_workers, 
            pin_memory=True,
            batch_sampler=self.datasampler(
                dataset=dataset, 
                shuffle=False,
                classes=self.val_tasks_split[idx],
                batch_size=self.batch_size,
                ) if self.datasampler is not None else None
            ) for idx, dataset in enumerate(self.test_datasets)
        ]

    def test_dataloader(self):
        full_classes = list(set(x for split in self.val_tasks_split for x in split))
        full_dataset = CLDataModule._get_subset(self.train_dataset, full_classes)  
        #ConcatDataset(self.train_datasets) # creates error / no easy way to get targets from this dataset
        
        return DataLoader(
            full_dataset, 
            batch_size=self.batch_size if self.datasampler is None else 1, 
            num_workers=self.test_val_num_workers,
            pin_memory=True,
            batch_sampler=self.datasampler(
                dataset=full_dataset, 
                shuffle=False,
                classes=full_classes,
                batch_size=self.batch_size,
            ) if self.datasampler is not None else None
        )

    def transform_targets(self, model, dream_target, task_index):
        if self.steps_to_locate_mean is not None:
            if not self.calculated_mean_std:
                self.std, self.mean = self.__calculate_std_mean_multidim(model=model, task_index=task_index)
            return self.tasks_processing_f(target=dream_target, model=model, mean=self.mean, std=self.std)

        return self.tasks_processing_f(target=dream_target, model=model)

    def __calculate_std_mean_multidim(self, model: base.CLBase, task_index):
        """
            Calculate mean and std from current train_task dataset.
            Returns a tuple of std and mean (in that order!) in a shape (<class <mean>>, <class <std>>)
        """
        normal_loader = DataLoader(
            self.train_task,
            batch_size=self.batch_size if self.datasampler is None else 1,
            num_workers=self.num_workers,
            shuffle=shuffle if self.datasampler is None else False, 
            pin_memory=True,
            batch_sampler=self.datasampler(
                    dataset=self.train_task, 
                    shuffle=self.shuffle,
                    classes=self.val_tasks_split[task_index],
                    batch_size=self.batch_size,
                ) if self.datasampler is not None else None
        )

        model_state_before = model.training

        if isinstance(self.steps_to_locate_mean, int):
            steps_to_locate_mean = self.steps_to_locate_mean
        elif isinstance(self.steps_to_locate_mean, float):
            steps_to_locate_mean = len(self.train_task) * self.steps_to_locate_mean
        else:
            raise Exception(f"Wrong type steps_to_locate_mean {type(self.steps_to_locate_mean)}. Accepted types are: int, float.")

        dataset_targets = get_target_from_dataset(self.train_task, toTensor=True)

        classes = torch.unique(dataset_targets).to(model.device)
        #model_output_size = model(model.get_objective_target()).size(dim=1)
        #classes_buffer = torch.zeros((len(classes), steps_to_locate_mean, 10),
        #    device=model.device,
        #    dtype=torch.float64)

        classes_buffer = []
        for _ in classes:
            classes_buffer.append([])

        #classes_buffer = np.zeros()
        #print(classes_buffer)
        #exit()

        if model_state_before:
            model.eval()

        with torch.no_grad():
            output_dim = None
            for idx, data in enumerate(normal_loader): # the majoritarian class does not matter here in batch
                if idx == steps_to_locate_mean:
                    break
                inputs, targets = data

                inputs = inputs.to(model.device)
                targets = targets.to(model.device)
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                if output_dim is None:
                    output_dim = len(outputs[0])
                for current_class in classes:
                    boolean_idxs = (targets == current_class)
                    current_idxs = torch.squeeze(torch.transpose(torch.nonzero(boolean_idxs), 0, 1))
                    #torch.stack()
                    #classes_buffer[current_class, idx] = torch.index_select(outputs, dim=0, index=current_idxs)
                    #torch.cat((classes_buffer[current_class], torch.index_select(outputs, dim=0, index=current_idxs)), dim=0)
                    classes_buffer[current_class].append(torch.index_select(outputs, dim=0, index=current_idxs))
            
            new_classes_buffer = torch.empty((len(classes), steps_to_locate_mean, output_dim), device=model.device)
            for idx, cl in enumerate(classes):
                torch.cat(classes_buffer[cl], dim=0, out=new_classes_buffer[idx])

            #print(new_classes_buffer.size())
            #exit()

            model.train(model_state_before)

            tmp = torch.std_mean(new_classes_buffer, dim=1) # reduce dim = 1 (batch)
            #print(tmp)
            return tmp

