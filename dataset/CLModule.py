import torch
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.trainer.supporters import CombinedLoader
from utils import data_manipulation as datMan
from utils.functional import target_processing
from dataset import dream_sets
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
#from lucent.optvis import param, render
from dream.custom_render import RenderVisState, render_vis, empty_loss_f
import wandb

import numpy as np
from math import ceil
from abc import ABC, abstractmethod

from utils.data_manipulation import get_target_from_dataset
from model import base
from utils import utils


class BaseCLDataModule(LightningDataModule, ABC):
    @abstractmethod
    def setup_tasks(self) -> None:
        """
            Create a list of train and test datasets, where the indices 
            indicate the task that will be processed in the future.
        """
        pass

    @abstractmethod
    def setup_task_index(self, task_index: int, loop_index: int) -> None:
        """
            Set the datasets that will be currently processed.
        """
        pass

    @abstractmethod
    def next(self) -> None:
        """
            Set the next that will be processed.
        """
        pass

    def generate_synthetic_data(self, model: LightningModule, task_index: int, layer_loss_obj=None) -> None:
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
        dream_image_f,
        target_processing_f,
        dreaming_batch_size=8,
        max_logged_dreams_per_target=8, 
        fast_dev_run:bool=False,
        fast_dev_run_dream_threshold=None,
        image_size = (3, 32, 32),
        progress_bar = None,
        empty_dream_dataset=None,
        optimizer=None,
        logger=None,
        render_transforms=None,
        dataset_class_labels=None,
        custom_f_steps=(0,),
        custom_f=lambda *args: None,
        enable_dream_transforms:bool=True,
        train_only_dream_batch_at:bool=False,
        richbar_refresh_fequency:int=50,
        standard_image_size=None,
        data_passer=None,
        dream_decorrelate=True,
    ):
        """
        Args:
            task_split: list containing list of class indices per task
            select_dream_tasks_f: function f(tasks:list, task_index:int) that will be used to select 
                the tasks to use during dreaming. If function uses more parameters, use lambda expression as adapter. 
            dream_objective_f: function f(target: list, model: model.base.CLBase, <source CLDataModule object, self>)
            target_processing_f: function that will take the list of targets and process them to the desired form.
                For example it will take a task and transform it to the point from normal distribution.
                The default function do nothing to targets. 
        """
        super().__init__()
        self.train_tasks_split = train_tasks_split
        self.select_dream_tasks_f = select_dream_tasks_f
        self.target_processing_f = target_processing_f
        self.dreams_per_target = dreams_per_target
        self.dreaming_batch_size = dreaming_batch_size
        self.dream_objective_f = dream_objective_f
        self.max_logged_dreams_per_target = max_logged_dreams_per_target
        self.progress_bar = progress_bar
        self.enable_dream_transforms = enable_dream_transforms

        self.dreams_dataset = empty_dream_dataset
        self.calculated_mean_std = False
        self.dream_decorrelate = dream_decorrelate

        self.fast_dev_run_dream_threshold = fast_dev_run_dream_threshold if isinstance(fast_dev_run_dream_threshold, tuple) or isinstance(fast_dev_run_dream_threshold, list) else (fast_dev_run_dream_threshold, )
        self.dream_threshold = dream_threshold if isinstance(dream_threshold, tuple) or isinstance(dream_threshold, list) else (dream_threshold, )
        self.fast_dev_run = fast_dev_run
        self.optimizer = optimizer
        self.logger = logger
        self.wandb_dream_img_table = wandb.Table(columns=['target', 'label', 'sample', 'image']) if dataset_class_labels is not None else wandb.Table(columns=['target', 'sample', 'image'])
        self.render_transforms = render_transforms
        self.dream_image = dream_image_f(image_size=image_size, dreaming_batch_size=dreaming_batch_size, decorrelate=dream_decorrelate)
        
        self.dataset_class_labels = dataset_class_labels
        self.custom_f_steps = custom_f_steps
        self.custom_f = custom_f
        self.train_only_dream_batch_at = train_only_dream_batch_at
        self.richbar_refresh_fequency = richbar_refresh_fequency
        self.standard_image_size = standard_image_size if standard_image_size is not None else image_size
        self.data_passer = data_passer

        print(f"Train task split: {self.train_tasks_split}")

        if(empty_dream_dataset is None):
            raise Exception("Empty dream dataset.")

        self.wandb_flushed = False

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

    def flush_wandb(self):
        wandb.log({'train_dream_examples': self.wandb_dream_img_table})
        self.wandb_flushed = True

    def __del__(self):
        if not (self.wandb_flushed):
            print('WARNING:\tdreaming images were not flushed by wandb.')

    def generate_synthetic_data(self, model: LightningModule, task_index: int, layer_loss_obj=None) -> None:
        """Generate new dreams."""
        primal_dream_targets = self.select_dream_tasks_f(self.train_tasks_split, task_index)
        #dream_targets = self.transform_targets(model=model, dream_targets=primal_dream_targets, task_index=task_index)
        dream_targets = primal_dream_targets

        model_mode = model.training # from torch.nn.Module
        if model_mode:
            model.eval()

        name = f"run_{task_index}"
        run_name = [name, name]
        custom_loss_gather_f = layer_loss_obj.gather_loss if hasattr(layer_loss_obj, 'gather_loss') else empty_loss_f
        if(self.logger is not None and custom_loss_gather_f is not None):
            custom_loss_gather_f = utils.log_wandb_tensor_decorator(custom_loss_gather_f, run_name, self.logger)

        thresholds = self.fast_dev_run_dream_threshold if self.fast_dev_run and self.fast_dev_run_dream_threshold is not None else self.dream_threshold
        self.render_vis_display_additional_info = True
        rendervis_state = RenderVisState(
                model=model,
                optim_image=self.dream_image,
                custom_f_steps=self.custom_f_steps,
                custom_f=self.custom_f,
                optimizer=self.optimizer,
                transforms=self.render_transforms,
                thresholds=thresholds,
                enable_transforms=self.enable_dream_transforms,
                standard_image_size=self.standard_image_size,
                custom_loss_gather_f=custom_loss_gather_f,
                display_additional_info=self.render_vis_display_additional_info,
                preprocess=False,
                device=model.device,
            )

        iterations = ceil(self.dreams_per_target / self.dreaming_batch_size)
        new_dreams, new_targets = self._generate_dreams(
            model=model,
            dream_targets=dream_targets, 
            iterations=iterations,
            task_index=task_index,
            layer_loss_obj=layer_loss_obj,
            rendervis_state=rendervis_state,
            run_name=run_name,
        )

        self.dreams_dataset.extend(new_dreams, new_targets, model)

        self.calculated_mean_std = False
        if model_mode:
            model.train()

        rendervis_state.unhook()

    def save_dream_dataset(self, location):
        self.dreams_dataset.save(location)
    
    def load_dream_dataset(self, location):
        self.dreams_dataset.load(location)

    def _generate_dreams(self, model, dream_targets, iterations, task_index, layer_loss_obj, rendervis_state, run_name:list[str]):
        new_dreams = []
        new_targets = []

        if (self.progress_bar is not None):
            self._generate_dreams_impl(
                model=model, 
                dream_targets=dream_targets, 
                iterations=iterations, 
                task_index=task_index,
                new_dreams=new_dreams, 
                new_targets=new_targets, 
                progress=self.progress_bar,
                layer_loss_obj=layer_loss_obj,
                rendervis_state=rendervis_state,
                run_name=run_name,
            )
            self.progress_bar.clear_dreaming()
        else:
            with Progress(
                "[progress.description]{task.description}",
                BarColumn(complete_style="magenta"),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
            ) as progress:
                self._generate_dreams_impl(
                    model=model, 
                    dream_targets=dream_targets, 
                    iterations=iterations, 
                    task_index=task_index,
                    new_dreams=new_dreams, 
                    new_targets=new_targets, 
                    progress=progress,
                    layer_loss_obj=layer_loss_obj,
                    rendervis_state=rendervis_state,
                    run_name=run_name,
                )
        new_dreams = torch.cat(new_dreams)
        new_targets = torch.tensor(new_targets)
        return new_dreams, new_targets

    def _generate_dreams_impl(self, model, dream_targets, iterations, task_index, new_dreams, new_targets, progress, layer_loss_obj, rendervis_state, run_name:list[str]):
        progress.setup_dreaming(dream_targets=dream_targets)
        for target in sorted(dream_targets):
            if(layer_loss_obj is not None):
                layer_loss_obj.set_current_class(target)
            run_name[0] = f"{run_name[1]}/target_{target}"
            target_dreams = self._generate_dreams_for_target(
                model=model, 
                target=target, 
                iterations=iterations, 
                progress_bar=progress,
                task_index=task_index,
                rendervis_state=rendervis_state,
            )
            new_targets.extend([target] * target_dreams.shape[0])
            new_dreams.append(target_dreams)
            progress.update_dreaming(0)
    
    def _log_target_dreams(self, new_dreams, target):
        if not self.fast_dev_run and self.logger is not None:
            for idx, image in enumerate(new_dreams):
                if(self.max_logged_dreams_per_target <= idx):
                    break
                img = wandb.Image(
                    image,
                    caption=f"sample: {idx} target: {target}",
                )
                if(self.dataset_class_labels is not None and target in self.dataset_class_labels):
                    self.wandb_dream_img_table.add_data(
                        target, 
                        self.dataset_class_labels[target],
                        idx, 
                        img
                    )
                else:
                    self.wandb_dream_img_table.add_data(
                        target, 
                        idx, 
                        img
                    )

    def _log_fast_dev_run(self, new_dreams, new_targets):
        if not self.fast_dev_run:
            num_dreams = new_dreams.shape[0]
            wandb.log(
                {
                    "examples": [
                        wandb.Image(
                            new_dreams[i, :, :, :],
                            caption=f"sample: {i} target: {new_targets[i]}",
                        )
                        for i in range(min(num_dreams, self.max_logged_dreams_per_target))
                    ]
                }
            )

    def _generate_dreams_for_target(self, model, target, iterations, progress_bar, task_index, rendervis_state):
        dreams = []
        progress_bar.setup_repeat(target=target, iterations=iterations)
        for _ in range(iterations):
            target_point = self.transform_targets(model=model, dream_target=target, task_index=task_index)
            objective = self.dream_objective_f(target=target, target_point=target_point, model=model, source_dataset_obj=self)
            rendervis_state.objective_f = objective

            numpy_render = torch.from_numpy(
                render_vis(
                    render_dataclass=rendervis_state,
                    show_image=False,
                    progress_bar=progress_bar,
                    refresh_fequency=self.richbar_refresh_fequency,
                )[-1] # return the last, most processed image (thresholds)
            ).detach()
            numpy_render = torch.permute(numpy_render, (0, 3, 1, 2))
            self.render_vis_display_additional_info = False

            dreams.append(numpy_render)
            progress_bar.update_dreaming(1)
        target_dreams = torch.cat(dreams)
        self._log_target_dreams(target_dreams, target)
        return target_dreams

    def transform_targets(self, model, dream_target, task_index) -> torch.Tensor:
        """
            Invoked after every new dreaming that produces an image.
            Override this if you need more than what the function target_processing_f itself can offer.
            Returns tensor representing target. It can be one number or an array of numbers (point)
        """
        return self.target_processing_f(target=dream_target, model=model)

    def get_task_classes(self, task_number):
        return self.train_tasks_split[task_number]

class CLDataModule(DreamDataModule):
    def __init__(
        self,
        dataset_class,
        data_transform,
        val_tasks_split=None,
        batch_size=32,
        dream_batch_size=8,
        num_workers=0,
        shuffle=True,
        steps_to_locate_mean = None,
        dream_num_workers=0,
        test_val_num_workers=0,
        dream_shuffle=True,
        datasampler=None,
        root="data",
        swap_datasets=False,
        use_dreams_at_start=False,
        dream_decorrelate=True,
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

        self.dream_num_workers = dream_num_workers
        self.dream_shuffle = dream_shuffle if dream_shuffle else shuffle
        self.test_val_num_workers = test_val_num_workers
        self.datasampler = datasampler if datasampler is not None else None
        self.root = root
        self.dream_batch_size = dream_batch_size

        self.train_task = None
        self.dream_dataset_for_current_task = None
        self.current_task_index = None
        self.current_loop_index = None
        # TODO
        self.dataset_class = dataset_class
        self.data_transform = data_transform
        self.swap_datasets = swap_datasets
        self.use_dreams_at_start = use_dreams_at_start

        print(f"Validation task split: {self.val_tasks_split}")
        if(self.swap_datasets):
            print(f"INFO: Datasets in swap_datasets mode.")

    def prepare_data(self):
        train_dataset = self.dataset_class(
            root=self.root, train=True, transform=self.data_transform, download=True
        )
        test_dataset = self.dataset_class(root=self.root, train=False, transform=self.data_transform)

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
            self.dream_dataset_for_current_task = self.dreams_dataset 
        else: 
            self.dream_dataset_for_current_task = None

    def setup_task_index(self, task_index: int=0, loop_index: int=0) -> None:
        """
            Choose the index of the task that will be used during training.
        """
        #TODO - to log
        if task_index >= len(self.train_datasets):
            raise Exception(f"Index {task_index} out of range {len(self.train_datasets)}")
        print(f"Selected task number: {task_index}")
        self.current_task_index = task_index
        self.current_loop_index = loop_index
        self.train_task = self.train_datasets[task_index]
        self.__setup_dream_dataset()

        if(self.train_task is None or len(self.train_task) <= 0):
            raise Exception(f"Train task dataset not set properly. Used index {task_index} from {len(self.train_datasets)}")

    def next(self) -> None:
        if(self.current_task_index is None):
            self.current_task_index = 0
            self.current_loop_index = 0
            return
        if self.current_task_index >= len(self.train_datasets):
            raise Exception(f"Index for train dataset out of range: '{len(self.train_datasets)}'")
        self.current_task_index += 1 
        self.current_loop_index += 1
        self.train_task = self.train_datasets[self.current_task_index]
        self.__setup_dream_dataset()
            
        if(self.train_task is None or len(self.train_task) <= 0):
            raise Exception(f"Train task dataset not set properly. Used index '{self.current_task_index}' from '{len(self.train_datasets)}'")

    def generate_synthetic_data(self, model: LightningModule, task_index: int, layer_loss_obj=None) -> None:
        super().generate_synthetic_data(model=model, task_index=task_index, layer_loss_obj=layer_loss_obj)
        self.__setup_dream_dataset()

    def _get_all_prev_dream_classes(self) -> list:
        to_range = self.current_task_index if len(self.train_tasks_split) >= self.current_task_index else len(self.train_tasks_split)
        dream_tasks_classes = set()
        for i in range(to_range + 1): # i < to_range
            dream_tasks_classes = dream_tasks_classes.union(self.train_tasks_split[i])
        if(len(dream_tasks_classes) == 0):
            raise Exception("Dream dataset selected classes are empty!")
        return list(dream_tasks_classes)

    def _select_dream_classes_for_task(self, class_list, index):
        if(len(class_list) <= index):
            ret = set()
            for i in class_list:
                ret = ret.union(i)
            return ret
        return class_list[index]

    def _check_dream_dataset_setup(self) -> bool:
        inner_check = bool(
            (self.swap_datasets and self.current_loop_index % 2 == 1) 
            or utils.check_python_index(self.train_only_dream_batch_at, self.data_passer['num_loops'], self.data_passer['current_loop']) 
            or (self.use_dreams_at_start and self.current_loop_index == 0)
        )
        if(self.dream_dataset_for_current_task is None and inner_check):
            raise Exception("Dream dataset not properly set. Try call before 'generate_synthetic_data' or 'next' or 'setup_task_index'")
        return bool(
            self.dream_dataset_for_current_task is not None and inner_check
        )

    def train_dataloader(self):
        """
            Returns the dictionary of :
            - "normal": normal_loader
            - "dream": dream_loader [Optional position]
            - "hidden_normal": optional normal_loader if option train_only_dream_batch_at is set.
        """
        if self.train_task is None: # check
            raise Exception("No task index set for training")
        dream_loader = None
        if self._check_dream_dataset_setup():
            # first loop is always False. After accumulating dreams this dataloader will be used.
            shuffle = self.dream_shuffle if self.datasampler is None else False
            # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
            dream_tasks_classes = self._get_all_prev_dream_classes()
            print(f"Selected dream dataloader classes: {dream_tasks_classes}. Use datasampler={self.datasampler is not None}")

            if(len(self.dreams_dataset) == 0):
                raise Exception("Empty dream dataset. Run dream generation or load dreams from file.")

            dream_loader = DataLoader(
                self.dreams_dataset, # give full dataset, var classes is used to select classes
                batch_size=self.batch_size if self.datasampler is None else 1, 
                num_workers=self.dream_num_workers, 
                shuffle=shuffle if self.datasampler is None else False, 
                pin_memory=True,
                batch_sampler=self.datasampler(
                    dataset=self.dreams_dataset , # give full dataset, var classes is used to select classes
                    shuffle=self.dream_shuffle,
                    classes=dream_tasks_classes, # all classes that were dream of at least once
                    batch_size=self.dream_batch_size,
                ) if self.datasampler is not None else None
            )

        normal_classes = self._select_dream_classes_for_task(self.val_tasks_split, self.current_task_index)

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
                    classes=normal_classes,
                    batch_size=self.batch_size,
                ) if self.datasampler is not None else None
        )

        normal_key = "normal"
        # need new label to not use this dataset but still keep CombinedLoader functionality
        if(utils.check_python_index(self.train_only_dream_batch_at, self.data_passer['num_loops'], self.data_passer['current_loop'])):
            normal_key = 'hidden_normal'
        else:
            print(f"Selected classes for normal dataloader: {normal_classes}")

        if dream_loader is not None:
            loaders = {normal_key: normal_loader, "dream": dream_loader}
        else:
            loaders = {normal_key: normal_loader}
        return CombinedLoader(loaders, mode="max_size_cycle")

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
        full_dataset = CLDataModule._get_subset(self.test_dataset, full_classes)  
        #ConcatDataset(self.train_datasets) # creates error / no easy way to get targets from this dataset

        print(f'Testing for classes: {full_classes}')
        
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

        #return DataLoader(
        #    full_dataset, 
        #    batch_size= 1, 
        #    num_workers=self.test_val_num_workers,
        #    pin_memory=True
        #)

    def transform_targets(self, model, dream_target, task_index):
        if self.steps_to_locate_mean is not None:
            if not self.calculated_mean_std:
                self.std, self.mean = self.__calculate_std_mean_multidim(model=model, task_index=task_index)
            return self.target_processing_f(target=dream_target, model=model, mean=self.mean, std=self.std)

        return self.target_processing_f(target=dream_target, model=model)

    def __calculate_std_mean_multidim(self, model: base.CLBase, task_index):
        """
            Calculate mean and std from current train_task dataset.
            Returns a tuple of std and mean (in that order!) in a shape (<class <mean>>, <class <std>>)
        """
        normal_loader = DataLoader(
            self.train_task,
            batch_size=self.batch_size if self.datasampler is None else 1,
            num_workers=self.num_workers,
            shuffle=self.shuffle if self.datasampler is None else False, 
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
                if isinstance(outputs, tuple) or isinstance(outputs, list):
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

