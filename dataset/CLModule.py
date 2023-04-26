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
from utils import utils, setup_args
from utils.functional.model_optimizer import ModelOptimizerManager
from dataclasses import dataclass, field
from typing import Union
from argparse import Namespace


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

    def generate_synthetic_data(self, model: LightningModule, task_index: int, layer_hook_obj=None, input_image_train_after_obj=None) -> None:
        """
            Generate new dreams from the tasks.
        """
        pass

class DreamDataModule(BaseCLDataModule, ABC):
    @dataclass
    class Config():
        dataset_labels: dict = None
        train_tasks_split: list = None
        custom_f_steps: list = (0,)
        richbar_refresh_fequency: int = 50

    @dataclass
    class Visualization():
        per_target: int
        threshold: list
        image_type: str = 'fft'
        batch_size: int = 8
        disable_transforms: bool = False
        only_vis_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
        image_size: list = (3, 32, 32)
        standard_image_size: Union[int, list, None] = None
        decorrelate: bool = True
        max_logged_image_per_target: int = 8

        def __post_init__(self):
            self.transforms = not self.disable_transforms
            self.threshold = self.threshold if isinstance(self.threshold, tuple) or isinstance(self.threshold, list) else (self.threshold, )
            self.standard_image_size = self.standard_image_size if self.standard_image_size is not None else self.image_size

        @dataclass
        class Optim():
            lr: float = 1e-3
            dtype: str = 'adam'
        
    CONFIG_MAP = {
        'vis': Visualization,
        'vis_optim': Visualization.Optim,
        'cfg': Config
    }

    VAR_MAP = {
        'config': 'cfg',
        'vis': 'cfg_vis',
        'vis.optim': 'cfg_optim',
    }

    def __init__(self,
        select_dream_tasks_f,
        dream_objective_f,
        dream_image_f,
        target_processing_f,
        fast_dev_run:bool=False,
        fast_dev_run_dream_threshold=None,
        progress_bar=None,
        empty_dream_dataset=None,
        logger=None,
        render_transforms=None,
        custom_f=lambda *args: None,
        data_passer=None,
        args=None,
        cfg_map=None,
        var_map=None,
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
        self._map_cfg(args=args, cfg_map=cfg_map, var_map=var_map)

        self.select_dream_tasks_f = select_dream_tasks_f
        self.target_processing_f = target_processing_f
        self.dream_objective_f = dream_objective_f
        self.progress_bar = progress_bar

        self.dreams_dataset = empty_dream_dataset
        self.calculated_mean_std = False

        self.fast_dev_run_dream_threshold = fast_dev_run_dream_threshold if isinstance(fast_dev_run_dream_threshold, tuple) or isinstance(fast_dev_run_dream_threshold, list) else (fast_dev_run_dream_threshold, )
        self.fast_dev_run = fast_dev_run
        self.logger = logger
        
        self.wandb_dream_img_table = wandb.Table(columns=['target', 'label', 'sample', 'image']) if self.cfg.dataset_labels is not None else wandb.Table(columns=['target', 'sample', 'image'])
        self.render_transforms = render_transforms
        self.dream_image = dream_image_f(
            dtype=self.cfg_vis.image_type, 
            image_size=self.cfg_vis.image_size, 
            dreaming_batch_size=self.cfg_vis.batch_size, 
            decorrelate=self.cfg_vis.decorrelate
        )
        
        self.custom_f = custom_f
        self.data_passer = data_passer

        print(f"Train task split: {self.cfg.train_tasks_split}")

        if(empty_dream_dataset is None):
            raise Exception("Empty dream dataset.")

        self.wandb_flushed = False

    def _map_from_args(self, args, not_from):
        self.cfg=self.CONFIG_MAP['cfg'](
            **utils.get_obj_dict(args.datamodule, not_from)
        )

        self.cfg_vis=self.CONFIG_MAP['vis'](
            **utils.get_obj_dict(args.datamodule.vis, not_from)
        )

        self.cfg_vis_optim=self.CONFIG_MAP['vis_optim'](
            **utils.get_obj_dict(args.datamodule.vis.optim, not_from)
        )

    def _map_cfg(self, args, cfg_map, var_map):
        setup_args.check(self, args, cfg_map)

        not_from = Namespace
        self._map_from_args(args, not_from)
        setup_args.setup_map(self, args, cfg_map, var_map)

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

    def get_rendervis(self, model, custom_loss_gather_f, input_image_train_after_hook=None):
        thresholds = self.fast_dev_run_dream_threshold if self.fast_dev_run and self.fast_dev_run_dream_threshold is not None else self.cfg_vis.threshold
        optimizer = ModelOptimizerManager(optimizer_type=self.cfg_vis_optim.dtype).get_optimizer(**vars(self.cfg_vis_optim))
        return RenderVisState(
                model=model,
                optim_image=self.dream_image,
                custom_f_steps=self.cfg.custom_f_steps,
                custom_f=self.custom_f,
                optimizer=optimizer,
                transforms=self.render_transforms,
                thresholds=thresholds,
                enable_transforms=self.cfg_vis.transforms,
                standard_image_size=self.cfg_vis.standard_image_size,
                custom_loss_gather_f=custom_loss_gather_f,
                display_additional_info=True,
                preprocess=False,
                device=model.device,
                input_image_train_after_hook=input_image_train_after_hook
            )

    def get_custom_loss_gather_f(self, task_index, *args) -> tuple:
        gather_to_invoke = []
        name = f"dream_loss/run_{task_index}"
        run_name = [name, name]
        for arg in args:
            if(arg is not None):
                    if(hasattr(arg, 'gather_loss')):
                        gather_to_invoke.append(arg.gather_loss)
        def inner_custom_loss_gather_f(loss):
            for l in gather_to_invoke:
                loss = l(loss)
            return loss
        
        custom_loss_gather_f = utils.log_wandb_tensor_decorator(inner_custom_loss_gather_f, run_name, self.logger) if(self.logger is not None) else inner_custom_loss_gather_f
        return custom_loss_gather_f, run_name        

    def generate_synthetic_data(self, model: LightningModule, task_index: int, layer_hook_obj:list=None, input_image_train_after_obj:list=None) -> None:
        """Generate new dreams."""
        dream_targets = self.select_dream_tasks_f(self.cfg.train_tasks_split, task_index)

        model_mode = model.training # from torch.nn.Module
        if model_mode:
            model.eval()

        layer_hook_obj = layer_hook_obj if layer_hook_obj is not None else ()
        input_image_train_after_obj = input_image_train_after_obj if input_image_train_after_obj is not None else ()
        custom_loss_gather_f, run_name = self.get_custom_loss_gather_f(task_index, *layer_hook_obj, *input_image_train_after_obj)
        rendervis_state = self.get_rendervis(model=model, custom_loss_gather_f=custom_loss_gather_f, input_image_train_after_hook=input_image_train_after_obj)
        
        iterations = ceil(self.cfg_vis.per_target / self.cfg_vis.batch_size)

        new_dreams, new_targets = self._generate_dreams(
            model=model,
            dream_targets=dream_targets, 
            iterations=iterations,
            task_index=task_index,
            layer_hook_obj=layer_hook_obj,
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

    def _generate_dreams(self, model, dream_targets, iterations, task_index, layer_hook_obj:list, rendervis_state, run_name:list[str]):
        new_dreams = []
        new_targets = []

        self._generate_dreams_impl(
            model=model, 
            dream_targets=dream_targets, 
            iterations=iterations, 
            task_index=task_index,
            new_dreams=new_dreams, 
            new_targets=new_targets, 
            layer_hook_obj=layer_hook_obj,
            rendervis_state=rendervis_state,
            run_name=run_name,
        )
        if(self.progress_bar is not None):
            self.progress_bar.clear_dreaming()
        new_dreams = torch.cat(new_dreams)
        new_targets = torch.tensor(new_targets)
        return new_dreams, new_targets

    def _generate_dreams_impl(self, model, dream_targets, iterations, task_index, new_dreams, new_targets, layer_hook_obj:list, rendervis_state, run_name:list[str]):
        if(self.progress_bar is not None):
            self.progress_bar.setup_dreaming(dream_targets=dream_targets)
        for target in sorted(dream_targets):
            if(layer_hook_obj is not None and len(layer_hook_obj) != 0):
                for l in layer_hook_obj:
                    l.set_current_class(target)
            run_name[0] = f"{run_name[1]}/target_{target}"
            target_dreams = self.generate_dreams_for_target(
                model=model, 
                target=target, 
                iterations=iterations, 
                task_index=task_index,
                rendervis_state=rendervis_state,
            )
            new_targets.extend([target] * target_dreams.shape[0])
            new_dreams.append(target_dreams)
            if(self.progress_bar is not None):
                self.progress_bar.update_dreaming(0)
    
    def _log_target_dreams(self, new_dreams, target):
        if not self.fast_dev_run and self.logger is not None:
            for idx, image in enumerate(new_dreams):
                if(self.cfg_vis.max_logged_image_per_target <= idx):
                    break
                img = wandb.Image(
                    image,
                    caption=f"sample: {idx} target: {target}",
                )
                if(self.cfg.dataset_labels is not None and target in self.cfg.dataset_labels):
                    self.wandb_dream_img_table.add_data(
                        target, 
                        self.cfg.dataset_labels[target],
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
                        for i in range(min(num_dreams, self.cfg_vis.max_logged_image_per_target))
                    ]
                }
            )

    def generate_dreams_for_target(self, model, target, iterations, task_index, rendervis_state):
        dreams = []
        if(self.progress_bar is not None):
            self.progress_bar.setup_repeat(target=target, iterations=iterations)
        for _ in range(iterations):
            target_point = self.transform_targets(model=model, dream_target=target, task_index=task_index)
            rendervis_state.objective_f = self.dream_objective_f(target=target, target_point=target_point, model=model, source_dataset_obj=self)

            numpy_render = torch.from_numpy(
                render_vis(
                    render_dataclass=rendervis_state,
                    show_image=False,
                    progress_bar=self.progress_bar,
                    refresh_fequency=self.cfg.richbar_refresh_fequency,
                )[-1] # return the last, most processed image (thresholds)
            ).detach()
            numpy_render = torch.permute(numpy_render, (0, 3, 1, 2))
            rendervis_state.display_additional_info = False

            dreams.append(numpy_render)
            if(self.progress_bar is not None):
                self.progress_bar.update_dreaming(1)
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
        return self.cfg.train_tasks_split[task_number]

class CLDataModule(DreamDataModule):
    @dataclass
    class Visualization(DreamDataModule.Visualization):
        num_workers: int = None
        disable_shuffle: bool = False

        def __post_init__(self):
            super().__post_init__()
            self.shuffle = not self.disable_shuffle

        def post_init_CLDataModule(self, shuffle, num_workers):
            self.num_workers = self.num_workers if self.num_workers is not None else num_workers
            self.shuffle = self.shuffle if self.shuffle is not None else shuffle

    @dataclass
    class Config(DreamDataModule.Config):
        batch_size: int = 32
        disable_shuffle: bool = False
        num_workers: int = 4
        test_num_workers: int = None
        test_batch_size: int = None
        val_num_workers: int = None
        val_batch_size: int = None
        val_shuffle: bool = False
        test_shuffle: bool = False
        shuffle: bool = field(init=False, default=None)
        val_tasks_split: list = None
        root: str = "data"

        def __post_init__(self):
            self.test_num_workers = self.test_num_workers if self.test_num_workers is not None else self.num_workers
            self.val_num_workers = self.val_num_workers if self.val_num_workers is not None else self.num_workers
            self.test_batch_size = self.test_batch_size if self.test_batch_size is not None else self.batch_size
            self.val_batch_size = self.val_batch_size if self.val_batch_size is not None else self.batch_size

            self.val_tasks_split = self.val_tasks_split if self.val_tasks_split is not None else self.train_tasks_split

            self.shuffle = not self.disable_shuffle

    CONFIG_MAP = {
        'vis': Visualization,
        'vis_optim': Visualization.Optim,
        'cfg': Config,
    }

    def __init__(
        self,
        dataset_class,
        data_transform,
        steps_to_locate_mean = None,
        datasampler=None,
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
        self.cfg_vis.post_init_CLDataModule(self.cfg.shuffle, self.cfg.num_workers)

        self.steps_to_locate_mean = steps_to_locate_mean
        
        self.datasampler = datasampler if datasampler is not None else None

        self.train_task = None
        self.dream_dataset_for_current_task = None
        self.current_task_index = None
        self.current_loop_index = None
        # TODO
        self.dataset_class = dataset_class
        self.data_transform = data_transform

        print(f"Validation task split: {self.cfg.val_tasks_split}")        

    def prepare_data(self):
        train_dataset = self.dataset_class(
            root=self.cfg.root, train=True, transform=self.data_transform, download=True
        )
        test_dataset = self.dataset_class(root=self.cfg.root, train=False, transform=self.data_transform)

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.setup_tasks()

    def setup_tasks(self):
        self.train_datasets = self._split_dataset(
            self.train_dataset, self.cfg.train_tasks_split
        )
        self.test_datasets = self._split_dataset(
            self.test_dataset, self.cfg.val_tasks_split
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

    def generate_synthetic_data(self, model: LightningModule, task_index: int, layer_hook_obj=None, input_image_train_after_obj=None) -> None:
        super().generate_synthetic_data(model=model, task_index=task_index, layer_hook_obj=layer_hook_obj)
        self.__setup_dream_dataset()

    def _get_all_prev_dream_classes(self) -> list:
        to_range = self.current_task_index if len(self.cfg.train_tasks_split) >= self.current_task_index else len(self.cfg.train_tasks_split)
        dream_tasks_classes = set()
        for i in range(to_range + 1): # i < to_range
            dream_tasks_classes = dream_tasks_classes.union(self.cfg.train_tasks_split[i])
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
            utils.check_python_index(self.cfg_vis.only_vis_at, self.data_passer['num_loops'], self.data_passer['current_loop'])
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
            - "hidden_normal": optional normal_loader if option cfg_vis.only_vis_at is set.
        """
        if self.train_task is None: # check
            raise Exception("No task index set for training")
        dream_loader = None
        if self._check_dream_dataset_setup():
            # first loop is always False. After accumulating dreams this dataloader will be used.
            # batch_sampler option is mutually exclusive with batch_size, shuffle, sampler, and drop_last
            dream_tasks_classes = self._get_all_prev_dream_classes()
            print(f"Selected dream dataloader classes: {dream_tasks_classes}. Use datasampler={self.datasampler is not None}")

            if(len(self.dreams_dataset) == 0):
                raise Exception("Empty dream dataset. Run dream generation or load dreams from file.")

            dream_loader = DataLoader(
                self.dreams_dataset, # give full dataset, var classes is used to select classes
                batch_size=self.cfg_vis.batch_size if self.datasampler is None else 1, 
                num_workers=self.cfg_vis.num_workers, 
                shuffle=self.cfg_vis.shuffle if self.datasampler is None else False, 
                pin_memory=True,
                batch_sampler=self.datasampler(
                    dataset=self.dreams_dataset , # give full dataset, var classes is used to select classes
                    shuffle=self.cfg_vis.shuffle,
                    classes=dream_tasks_classes, # all classes that were dream of at least once
                    batch_size=self.cfg_vis.batch_size,
                ) if self.datasampler is not None else None
            )

        normal_classes = self._select_dream_classes_for_task(self.cfg.val_tasks_split, self.current_task_index)

        normal_loader = DataLoader(
            self.train_task,
            batch_size=self.cfg.batch_size if self.datasampler is None else 1,
            num_workers=self.cfg.num_workers,
            shuffle=self.cfg.shuffle if self.datasampler is None else False, 
            pin_memory=True,
            batch_sampler=self.datasampler(
                    dataset=self.train_task, 
                    shuffle=self.cfg.shuffle,
                    classes=normal_classes,
                    batch_size=self.cfg.batch_size,
                ) if self.datasampler is not None else None
        )

        normal_key = "normal"
        # need new label to not use this dataset but still keep CombinedLoader functionality
        if(utils.check_python_index(self.cfg_vis.only_vis_at, self.data_passer['num_loops'], self.data_passer['current_loop'])):
            normal_key = 'hidden_normal'
            print("INFO: Normal dataloader is hidden.")
        else:
            print(f"Selected classes for normal dataloader: {normal_classes}")

        if dream_loader is not None:
            print("INFO: Use dream dataloader.")
            loaders = {normal_key: normal_loader, "dream": dream_loader}
        else:
            loaders = {normal_key: normal_loader}
        return CombinedLoader(loaders, mode="max_size_cycle")

    def clear_dreams_dataset(self):
        self.dreams_dataset.clear()

    def val_dataloader(self):
        # must return multiple for multiple tasks (each for one task)
        return [
            DataLoader(dataset, 
            batch_size=self.cfg.val_batch_size if self.datasampler is None else 1, 
            num_workers=self.cfg.val_num_workers, 
            pin_memory=True,
            batch_sampler=self.datasampler(
                dataset=dataset, 
                shuffle=self.cfg.val_shuffle,
                classes=self.cfg.val_tasks_split[idx],
                batch_size=self.cfg.val_batch_size,
                ) if self.datasampler is not None else None
            ) for idx, dataset in enumerate(self.test_datasets)
        ]

    def test_dataloader(self):
        full_classes = list(set(x for split in self.cfg.val_tasks_split for x in split))
        full_dataset = CLDataModule._get_subset(self.test_dataset, full_classes)  
        #ConcatDataset(self.train_datasets) # creates error / no easy way to get targets from this dataset

        print(f'Testing for classes: {full_classes}')
        
        return DataLoader(
            full_dataset, 
            batch_size=self.cfg.test_batch_size if self.datasampler is None else 1, 
            num_workers=self.cfg.test_num_workers,
            pin_memory=True,
            batch_sampler=self.datasampler(
                dataset=full_dataset, 
                shuffle=self.cfg.test_shuffle,
                classes=full_classes,
                batch_size=self.cfg.test_batch_size,
            ) if self.datasampler is not None else None
        )

    def __calculate_std_mean_multidim(self, model: base.CLBase, task_index):
        """
            Calculate mean and std from current train_task dataset.
            Returns a tuple of std and mean (in that order!) in a shape (<class <mean>>, <class <std>>)
        """
        normal_loader = DataLoader(
            self.train_task,
            batch_size=self.cfg.batch_size if self.datasampler is None else 1,
            num_workers=self.cfg.num_workers,
            shuffle=self.cfg.shuffle if self.datasampler is None else False, 
            pin_memory=True,
            batch_sampler=self.datasampler(
                    dataset=self.train_task, 
                    shuffle=self.cfg.shuffle,
                    classes=self.cfg.val_tasks_split[task_index],
                    batch_size=self.cfg.batch_size,
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
        #model_output_size = model(model.get_objective_target_name()).size(dim=1)
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

