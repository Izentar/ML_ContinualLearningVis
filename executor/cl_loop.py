# Copyright The PyTorch Lightning team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Based on
# https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pl_examples/loop_examples/kfold.py
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pytorch_lightning.loops.base import Loop
from pytorch_lightning.loops.fit_loop import FitLoop
from pytorch_lightning.trainer.states import TrainerFn
from dataset.CLModule import BaseCLDataModule, CLDataModule

from config.default import fast_dev_run_config
import gc
import torch
import datetime, glob

from config.default import default_export_path, model_to_save_file_type
from model.statistics import base as layer_stat_framework
from loss_function import layerloss
from pytorch_lightning.trainer.supporters import CombinedLoader
from utils import utils
from model.statistics.base import ModelLayerStatistics, unhook
from collections.abc import Sequence
from loss_function import image_regularization

from dataclasses import dataclass, field
from argparse import Namespace
from utils import setup_args
import wandb
from utils import pretty_print as pp
import os
import time

########################################################################
#                     Here is the `Pseudo Code` for the base Loop.     #
# class Loop:                                                          #
#                                                                      #
#   def run(self, ...):                                                #
#       self.reset(...)                                                #
#       self.on_run_start(...)                                         #
#                                                                      #
#        while not self.done:                                          #
#            self.on_advance_start(...)                                #
#            self.advance(...)                                         #
#            self.on_advance_end(...)                                  #
#                                                                      #
#        return self.on_run_end(...)                                   #
########################################################################

class CLLoop(Loop):
    @dataclass
    class Config():
        plan: list[list[int]]
        schedule: list[int]
        num_tasks: Union[int, None] = field(default=None, init=False)
        num_loops: Union[int, None] = None
        train_at: Union[str, bool, int, list[str], list[bool], list[int], None] = True
        weight_reset_sanity_check: bool = False

        def __post_init__(self):
            self.num_tasks = len(self.plan)
            self.num_loops = self.num_tasks if self.num_loops is None else self.num_loops

    @dataclass
    class Visualization():
        generate_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
        clear_dataset_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
        measure_time: bool = True

        @dataclass
        class LayerLoss():
            @dataclass
            class MeanNorm():
                scaling: float = 0.01
                del_cov_after: bool =False
                use_at: Union[str, bool, int, list[str], list[bool], list[int], None] = None
                hook_to: Union[list[str], None] = False
                device: str = 'cuda'

            @dataclass
            class GradPruning():
                use_at: Union[str, bool, int, list[str], list[bool], list[int], None] = None
                percent: float = 0.01
                hook_to: Union[list[str], None] = False
                device: str = 'cuda'

            @dataclass
            class GradActivePruning():
                use_at: Union[str, bool, int, list[str], list[bool], list[int], None] = None
                percent: float = 0.01
                hook_to: Union[list[str], None] = False

            @dataclass
            class DeepInversion():
                use_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
                scale: float = 1e2
                hook_to: list[str] = None

                def __post_init__(self):
                    # only works for BatchNorm2d
                    self.hook_to = ['BatchNorm2d'] if self.hook_to is None else self.hook_to

        @dataclass
        class ImageRegularization():
            @dataclass
            class Variation():
                use_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
                scale: float = 2.5e-5

            @dataclass
            class L2():
                use_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
                coeff: float = 1e-05

    @dataclass
    class Model():
        reload_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
        reinit_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False

    @dataclass
    class Save():
        enable_checkpoint: bool = False
        model: bool = False
        dreams: bool = False
        layer_stats: bool = False
        root: str = None

        def __post_init__(self):
            if self.root is not None:
                self.root = Path(self.root)
            else:
                self.root = Path(default_export_path)

    @dataclass
    class Load():
        id: list = None
        model: bool = False
        dreams: bool = False
        layer_stats: bool = False
        root: str = None

        def __post_init__(self):
            if self.root is not None:
                self.root = Path(self.root)
            else:
                self.root = Path(default_export_path)
        
    @dataclass
    class LayerStats():
        use_at: Union[str, bool, int, list[str], list[bool], list[int], None] = None
        hook_to: Union[list[str], None] = False
        device: str = 'cuda'
        flush_to_disk: bool = False
        verbose: bool = False

    FILE_TYPE_MAP = {
        "dream": "dataset",
        "trained_model": "pt",
        "checkpoint": "ckp",
        "layer_stats": "stat",
    }

    def __init__(
        self,
        plan: list[list[int]],
        cfg_map: dict = None,
        args_map: dict = None,
        args = None,
        fast_dev_run_epochs=None,
        fast_dev_run=False,
        data_passer=None,
        weight_reset_sanity_check=False,
        data_module=None,
        progress_bar=None,
        folder_output_path=None,
    ) -> None:
        """
            plan: list of epoches per task
            save_export_path: save model parameters to path on current task

            num_loops - if None then the same as cfg.num_tasks. It is used to loop over the cfg.num_tasks.
                If num_loops == cfg.num_tasks then nothing is changed. 
                It can be used to loop more than the cfg.num_tasks after 
                all tasks are done.
        """
        super().__init__()
        self.CONFIG_MAP, self.VAR_MAP = self._get_config_maps()
        utils.check_cfg_var_maps(self)
        self._map_cfg(args=args, cfg_map=cfg_map, plan=plan, args_map=args_map)

        self.current_task: int = 0
        self.current_loop: int = 0
        self.previous_task:int = 0
        self.previous_loop:int = 0
        
        self.fast_dev_run_epochs = fast_dev_run_epochs
        self.fast_dev_run = fast_dev_run
        self.weight_reset_sanity_check = weight_reset_sanity_check
        self.data_module = data_module
        self.progress_bar = progress_bar

        self.folder_output_path = folder_output_path

        self.enable_data_parser = data_passer is not None
        self.data_passer = data_passer if data_passer is not None else {}
        self.custom_advance_f = None
        self.model_stats = None

        self._save_folder = None

        if(self.folder_output_path is None):
            self.folder_output_path = ""

        if utils.check_python_enabled(self.cfg_model.reinit_at) and utils.check_python_enabled(self.cfg_model.reload_at):
            raise Exception("ERROR: cfg_model.reinit_at and cfg_model.reload_at cannot be both true")

        for l in self.cfg.plan:
            if((not isinstance(l, Sequence)) or len(l) < 1):
                raise Exception(f'Bad plan: {self.cfg.plan}')

    def _get_config_maps(self):
        return {
            'cfg_vis': CLLoop.Visualization,
            'cfg_model': CLLoop.Model,
            'cfg_save': CLLoop.Save,
            'cfg_load': CLLoop.Load,
            'cfg_mean_norm': CLLoop.Visualization.LayerLoss.MeanNorm,
            'cfg_grad_pruning': CLLoop.Visualization.LayerLoss.GradPruning,
            'cfg_grad_activ_pruning': CLLoop.Visualization.LayerLoss.GradActivePruning,
            'cfg_deep_inversion': CLLoop.Visualization.LayerLoss.DeepInversion,
            'cfg_vis_regularization_var': CLLoop.Visualization.ImageRegularization.Variation,
            'cfg_vis_regularization_l2': CLLoop.Visualization.ImageRegularization.L2,
            'cfg_layer_stats': CLLoop.LayerStats,
            'cfg': CLLoop.Config,
        }, {
            '': 'cfg',
            'vis': 'cfg_vis',
            'model': 'cfg_model',
            'save': 'cfg_save',
            'load': 'cfg_load',
            'vis.layerloss.mean_norm': 'cfg_mean_norm',
            'vis.layerloss.grad_pruning': 'cfg_grad_pruning',
            'vis.layerloss.grad_activ_pruning': 'cfg_grad_activ_pruning',
            'vis.layerloss.deep_inversion': 'cfg_deep_inversion',
            'vis.image_reg.var': 'cfg_vis_regularization_var',
            'vis.image_reg.l2': 'cfg_vis_regularization_l2',
            'layer_stats': 'cfg_layer_stats',
        }

    def _map_cfg(self, args, cfg_map:dict, plan, args_map):
        setup_args.check(self, args, cfg_map)
        args = deepcopy(args)
        if(not hasattr(args.loop, 'plan')):
            setattr(args.loop, 'plan', plan)


        if(args is not None):
            args = setup_args.setup_args(args_map, args, 'loop')
            utils.setup_obj_dataclass_args(self, args=args, root_name='loop', recursive=True, recursive_types=[Namespace])
        setup_args.setup_cfg_map(self, args, cfg_map)

    @property
    def done(self) -> bool:
        return self.current_loop >= self.cfg.num_loops

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""
        self.current_task = 0
        self.current_loop = 0
        self.fit_loop.reset()

    def _update_data_passer(self):
        self.data_passer['current_task'] = self.current_task
        self.data_passer['current_loop'] = self.current_loop
        self.data_passer['num_tasks'] = self.cfg.num_tasks
        self.data_passer['num_loops'] = self.cfg.num_loops
        if(self.current_loop < self.cfg.num_loops):
            self.data_passer['epoch_per_task'] = self.epoch_num

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_tasks` from the `BaseCLDataModule` instance and store the
        original weights of the model."""
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)
        self._try_load_model()
        self.trainer.datamodule.setup_tasks()
        if(utils.check_python_enabled(self.cfg_model.reload_at)):
            # need deepcopy, because state_dict reference the tensor, not its copy
            self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())
            if(self.weight_reset_sanity_check):
                self.state_dict_sanity_check_val = self.trainer.lightning_module.model.get_objective_layer().weight.cpu()  
        if(utils.check_python_enabled(self.cfg_model.reinit_at)):
            if(self.weight_reset_sanity_check):
                self.state_dict_sanity_check_val = self.trainer.lightning_module.model.get_objective_layer().weight.cpu() 
        self._update_data_passer()
        self._try_load_dreams()

    def _gather_model_layer_stats_advance_f(self):
        pp.sprint(f"{pp.COLOR.NORMAL}hooking to model - {pp.COLOR.DETAIL}GATHER STATS{pp.COLOR.NORMAL} - task: {self.current_task}, loop {self.current_loop}")

        train_dataloader = self.data_module.train_dataloader()
        if(isinstance(train_dataloader, CombinedLoader)):
            train_dataloader = train_dataloader.loaders
        dataloader = train_dataloader.get('normal')
        if(dataloader is None):
            dataloader = train_dataloader.get('hidden_normal')
        self.model_stats, _ = layer_stat_framework.collect_model_layer_stats(
            model=self.trainer.lightning_module,
            single_dataloader=dataloader,
            device=self.cfg_layer_stats.device,
            hook_verbose=self.cfg_layer_stats.verbose,
            progress_bar=self.progress_bar,
            flush_to_disk=self.cfg_layer_stats.flush_to_disk,
            hook_to=self.cfg_layer_stats.hook_to,
            fast_dev_run=self.fast_dev_run,
        )
        self._try_save_model_layer_stats()

    def _try_generate_dream_print_msg(self, dtype:str):
        pp.sprint(f"{pp.COLOR.NORMAL}INFO: hooking model during visualization to - {pp.COLOR.DETAIL}{dtype}{pp.COLOR.NORMAL} - task: {self.current_task}, loop {self.current_loop}")

    def _try_generate_dream(self):
        if utils.check_python_index(self.cfg_vis.generate_at, self.cfg.num_loops, self.current_loop):
            layerloss_hook_obj = []
            layer_handles = []
            input_image_train_after_obj = []

            if(utils.check_python_index(self.cfg_mean_norm.use_at, self.cfg.num_loops, self.current_loop)):
                self._try_generate_dream_print_msg("LOSS FUNCTION")
                self._try_load_model_layer_stats()
                self.vis_layerloss_mean_norm = True
                tmp = layerloss.MeanNorm(device=self.cfg_mean_norm.device, del_cov_after=self.cfg_mean_norm.del_cov_after, scaling=self.cfg_mean_norm.scaling)
                layerloss_hook_obj.append(tmp)
                layer_handles.append(layer_stat_framework.hook_model_stats(
                    model=self.trainer.lightning_module, 
                    stats=self.model_stats, 
                    fun=tmp.hook_fun, 
                    hook_to=self.cfg_mean_norm.hook_to
                ))
            if(utils.check_python_index(self.cfg_grad_pruning.use_at, self.cfg.num_loops, self.current_loop)):
                self._try_generate_dream_print_msg("GRAD PRUNING FUNCTION")
                self._try_load_model_layer_stats()
                self.layerloss_grad_pruning = True
                tmp = layerloss.LayerGradPruning(device=self.cfg_grad_pruning.device, percent=self.cfg_grad_pruning.percent)
                layerloss_hook_obj.append(tmp)
                layer_handles.append(layer_stat_framework.hook_model_stats(
                    model=self.trainer.lightning_module, 
                    stats=self.model_stats, 
                    fun=tmp.hook_fun, 
                    hook_to=self.cfg_grad_pruning.hook_to
                ))
            if(utils.check_python_index(self.cfg_grad_activ_pruning.use_at, self.cfg.num_loops, self.current_loop)):
                self._try_generate_dream_print_msg("GRAD ACTIVE PRUNING FUNCTION")
                self._try_load_model_layer_stats()
                self.layerloss_grad_activ_pruning = True
                tmp = layerloss.LayerGradActivationPruning(percent=self.cfg_grad_activ_pruning.percent)
                layerloss_hook_obj.append(tmp)
                layer_handles.append(utils.hook_model(
                    model=self.trainer.lightning_module, 
                    fun=tmp.hook_fun, 
                    hook_to=self.cfg_grad_activ_pruning.hook_to
                ))

            if(utils.check_python_index(self.cfg_deep_inversion.use_at, self.cfg.num_loops, self.current_loop)):
                self._try_generate_dream_print_msg("DEEP INVERSION")
                tmp = layerloss.DeepInversionFeatureLoss(scale=self.cfg_deep_inversion.scale)
                self.layerloss_deep_inversion = True
                layerloss_hook_obj.append(tmp)
                layer_handles.append(utils.hook_model(
                    model=self.trainer.lightning_module, 
                    fun=tmp.hook_fun, 
                    hook_to=self.cfg_deep_inversion.hook_to
                ))

            if(utils.check_python_index(self.cfg_vis_regularization_var.use_at, self.cfg.num_loops, self.current_loop)):
                self._try_generate_dream_print_msg("VARIATION IMAGE REGULARIZATION")
                tmp = image_regularization.VariationRegularization(scale=self.cfg_vis_regularization_var.scale)
                input_image_train_after_obj.append(tmp)
                self.vis_regularization_var = True

            if(utils.check_python_index(self.cfg_vis_regularization_l2.use_at, self.cfg.num_loops, self.current_loop)):
                self._try_generate_dream_print_msg("L2 IMAGE REGULARIZATION")
                tmp = image_regularization.L2Regularization(coefficient=self.cfg_vis_regularization_l2.coeff)
                input_image_train_after_obj.append(tmp)
                self.vis_regularization_l2 = True
            
            pp.sprint(f"{pp.COLOR.NORMAL}DREAMING DURING TASK: {self.current_task}, loop {self.current_loop}")
            
            if(self.cfg_vis.measure_time):
                start = time.time()
            try:
                self.trainer.datamodule.generate_synthetic_data(
                    model=self.trainer.lightning_module, 
                    task_index=self.current_task, 
                    layer_hook_obj=layerloss_hook_obj,
                    input_image_train_after_obj=input_image_train_after_obj,
                )
            except KeyboardInterrupt:
                pp.sprint(f"{pp.COLOR.WARNING}Skipping visualization. Keyboard interruption.")
            if(self.cfg_vis.measure_time):
                end = time.time()
                hour, minutes, secs = utils.time_in_sec_format_to_hourly(end - start)
                pp.sprint(f"{pp.COLOR.NORMAL}Time generating features: {hour:02d}:{minutes:02d}:{secs:02d}")
            if(len(layer_handles) != 0):
                for l in layer_handles:
                    unhook(l)
            pp.sprint(f"{pp.COLOR.NORMAL}DREAMING END")

    def _model_weigth_sanity_check(self):
        if(self.weight_reset_sanity_check):
            test_sanity_val = self.trainer.lightning_module.model.get_objective_layer().weight.cpu()
            if torch.allclose(self.state_dict_sanity_check_val, test_sanity_val):
                pp.sprint(f"{pp.COLOR.WARNING}Model reset success")
                return
            pp.sprint(f'{pp.COLOR.WARNING}FAIL: Model reset fail sanity check.')
            pp.sprint(f"{pp.COLOR.WARNING}", self.state_dict_sanity_check_val)
            pp.sprint(f"{pp.COLOR.WARNING}", test_sanity_val)

    def _try_reset_model(self):
        # restore the original weights + optimizers and schedulers.
        if utils.check_python_index(self.cfg_model.reload_at, self.cfg.num_loops, self.current_loop):
            pp.sprint(f'{pp.COLOR.NORMAL}INFO: Model reloaded.')
            self.trainer.lightning_module.load_state_dict(
                self.lightning_module_state_dict, strict=True
            )
            self.trainer.strategy.setup_optimizers(self.trainer)
            self._model_weigth_sanity_check()
        if utils.check_python_index(self.cfg_model.reinit_at, self.cfg.num_loops, self.current_loop):
            pp.sprint(f'{pp.COLOR.NORMAL}INFO: Model reinit.')
            self.trainer.lightning_module.init_weights()
            self._model_weigth_sanity_check()

    @property
    def epoch_num(self):
        if(self.fast_dev_run and self.fast_dev_run_epochs is not None):
            return self.fast_dev_run_epochs
        if(len(self.cfg.plan[self.current_task]) <= self.current_loop):
            tmp = self.cfg.plan[self.current_task][-1]
            pp.sprint(f'{pp.COLOR.WARNING}WARNING: At loop {self.current_loop} selected last epoch per task "{tmp}" because list index out of range.')
            return tmp
        return self.cfg.plan[self.current_task][self.current_loop]

    @property
    def save_folder(self) -> str | None:
        if(self._save_folder is None):
            return self._generate_save_path('trained_model')[0]
        return self._save_folder

    def _setup_loop(self):
        self.replace(
            fit_loop=FitLoop(max_epochs=self.epoch_num)
        )
        if(utils.check_python_index(self.cfg_layer_stats.use_at, self.cfg.num_loops, self.current_loop)):
            pp.sprint(f"{pp.COLOR.NORMAL}INFO: HOOKING UP LOOP TO GATHER STATISTICS")
            self.custom_advance_f = self._gather_model_layer_stats_advance_f # run custon data gathering loop
            self.layer_stats_use_at = True
        elif(utils.check_python_index(self.cfg.train_at, self.cfg.num_loops, self.current_loop)):
            pp.sprint(f"{pp.COLOR.NORMAL}INFO: HOOKING UP NORMAL LOOP")
            self.custom_advance_f = self.fit_loop.run # run subloop - FitLoop
        else:
            self.custom_advance_f = lambda: pp.sprint(f"{pp.COLOR.WARNING}INFO: SKIPPING ANY TRAINING at loop {self.current_loop}")
    
    def _generate_save_path_dream(self) -> str:
        ret = Path('')
        ll = ""
        sufix = Path('')
        if(utils.check_python_enabled(self.cfg_mean_norm.use_at)):
            ll = "layerloss"
            sufix = sufix / "mean_norm"
        if(utils.check_python_enabled(self.cfg_grad_pruning.use_at)):
            ll = "layerloss"
            sufix = sufix / "grad_pruning"
        if(utils.check_python_enabled(self.cfg_grad_activ_pruning.use_at)):
            ll = "layerloss"
            sufix = sufix / "grad_activ_pruning"
        if(utils.check_python_enabled(self.cfg_deep_inversion.use_at)):
            ll = "layerloss"
            sufix = sufix / "cfg_deep_inversion"

        ret = ret / ll / sufix

        reg = ""
        sufix = Path('')

        if(utils.check_python_enabled(self.cfg_vis_regularization_var.use_at)):
            reg = "regularization"
            sufix = sufix / "var"
        if(utils.check_python_enabled(self.cfg_vis_regularization_l2.use_at)):
            reg = "regularization"
            sufix = sufix / "l2"

        ret = ret / reg / sufix

        return ret 

    def _generate_save_path(self, dtype: str):
        date = datetime.datetime.now().strftime("%d-%m-%Y")
        time = datetime.datetime.now().strftime("%H-%M-%S")
        model_type = type(self.trainer.lightning_module.model).__name__
        overlay_type = type(self.trainer.lightning_module).__name__

        # here put your root folder
        folder = Path(overlay_type) / model_type / self._generate_save_path_dream()

        adds = ""
        if(hasattr(self, 'layer_stats_use_at') and self.layer_stats_use_at and len(self.cfg_layer_stats.hook_to) != 0):
            adds = f"{adds}_layer_stat"
        if(hasattr(self, 'vis_layerloss_mean_norm') and self.vis_layerloss_mean_norm and len(self.cfg_mean_norm.hook_to) != 0):
            adds = f"{adds}_mean_norm"
        if(hasattr(self, 'layerloss_grad_pruning') and self.vis_layerloss_mean_norm and len(self.cfg_grad_pruning.hook_to) != 0):
            adds = f"{adds}_grad_pruning"
        if(hasattr(self, 'layerloss_grad_activ_pruning') and self.layerloss_grad_activ_pruning and len(self.cfg_grad_activ_pruning.hook_to) != 0):
            adds = f"{adds}_grad_activ_pruning"
        if(hasattr(self, 'layerloss_deep_inversion') and self.layerloss_deep_inversion and len(self.cfg_deep_inversion.hook_to) != 0):
            adds = f"{adds}_deep_inversion"
        
        folder = folder / adds
        adds = ""

        if(hasattr(self, 'vis_regularization_var') and self.vis_regularization_var and utils.check_python_enabled(self.cfg_vis_regularization_var.use_at)):
            adds = f"{adds}image_reg_var"
        if(hasattr(self, 'vis_regularization_l2') and self.vis_regularization_l2 and utils.check_python_enabled(self.cfg_vis_regularization_l2.use_at)):
            adds = f"{adds}_image_reg_l2"

        folder = folder / adds

        if(not hasattr(self, 'run_name') or self.run_name is None):
            self.run_name = f"d{date}_h{time}_{wandb.run.id}"
            pp.sprint(f"{pp.COLOR.NORMAL}INFO: Generated current run name: {self.run_name}")

        file_type = self.FILE_TYPE_MAP.get(dtype)
        if(not file_type):
            raise Exception(f'Not found in lookup map: {dtype}')
        
        gen_path = self.cfg_save.root / folder / self.folder_output_path / self.run_name
        Path.mkdir(gen_path, parents=True, exist_ok=True)
        self._save_folder = gen_path
        return gen_path, f"{dtype}.{date}_{time}_loop_{self.current_loop}_epoch_{self.epoch_num}.{file_type}"
    
    def _generate_load_path(self, dtype: str):
        if(self.cfg_load.id is None or len(self.cfg_load.id) == 0):
            raise Exception("Tried loading while id is None")
        file_type = self.FILE_TYPE_MAP.get(dtype)
        if(not file_type):
            raise Exception(f'Not found in lookup map: {dtype}')
        
        root = self.cfg_load.root
        for id in self.cfg_load.id:
            phrase = f"**/*{id}*"
            outer_path = glob.glob(pathname=phrase, root_dir=root, recursive=True)
            if(len(outer_path) != 1):
                raise Exception(f"Found {len(outer_path)} possible paths. Phrase ''{phrase}'', possible paths from: {outer_path}. Root: {root}")
            root = root / outer_path[0]

        if(os.path.isfile(root)):
            path = root
        else:
            phrase = f"**/*{dtype}*"
            path = glob.glob(pathname=phrase, root_dir=root, recursive=True)
            if(len(path) != 1):
                raise Exception(f"Found {len(path)} possible paths. Phrase ''{phrase}'' possible paths from: {path}. Root: {root}")
            path = root / path[0]
        return path

    def _try_save_dreams(self):
        if(self.cfg_save.dreams and not self.fast_dev_run):
            filepath, name = self._generate_save_path('dream')
            self.trainer.datamodule.save_dream_dataset(filepath / name)

    def _try_load_dreams(self):
        if(self.cfg_load.dreams):
            path = self._generate_load_path('dream')
            self.trainer.datamodule.load_dream_dataset(path)
            pp.sprint(f'{pp.COLOR.NORMAL}INFO: Loaded dreams from {path}')

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_task_index` from the `BaseCLDataModule` instance."""
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)

        if(utils.check_python_index(self.cfg_vis.clear_dataset_at, self.cfg.num_loops, self.current_loop)):
            pp.sprint(f'{pp.COLOR.WARNING}INFO: Dreams cleared at loop {self.current_loop}, task {self.current_task}')
            self.trainer.datamodule.clear_dreams_dataset()
        self._try_generate_dream()
        self._update_data_passer()
        pp.sprint(f"{pp.COLOR.NORMAL}STARTING TASK {self.current_task}, loop {self.current_loop} -- classes in task {self.trainer.datamodule.get_task_classes(self.current_task)}")
        
        self._try_reset_model()
        self.trainer.datamodule.setup_task_index(self.current_task, self.current_loop)
        self._setup_loop()
        
        # TODO I think there is no function named like that
        if callable(getattr(self.trainer.lightning_module, "on_task_start", None)):
            self.trainer.lightning_module.on_task_start()

    def _next(self) -> None:
        self.previous_task = self.current_task
        self.previous_loop = self.current_loop
        self.current_task += 1 if self.current_task < self.cfg.num_tasks - 1 else 0
        self.current_loop += 1
        self._update_data_passer()

    def _try_export_model_checkpoint(self):
        if self.cfg_save.enable_checkpoint and not self.fast_dev_run:
            filepath, name = self._generate_save_path('checkpoint')
            self.trainer.save_checkpoint(filepath / name)

    def _try_save_trained_model(self):
        if(self.cfg_save.model and not self.fast_dev_run):
            filepath, name = self._generate_save_path('trained_model')
            self.trainer.save_checkpoint(
                filepath / name,
                weights_only=True
            )

    def _try_load_model(self):
        if(self.cfg_load.model):
            path = self._generate_load_path('trained_model')
            checkpoint = torch.load(path)
            self.trainer.lightning_module.load_state_dict(checkpoint["state_dict"])
            pp.sprint(f'{pp.COLOR.NORMAL}INFO: Loaded model from "{path}"')

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        try:
            self.custom_advance_f(*args, **kwargs)
        except KeyboardInterrupt:
            pp.sprint(f"{pp.COLOR.WARNING}Skipping loop. Keyboard interruption.")

    def on_advance_end(self) -> None:
        """Used to save the weights of the current task and reset the LightningModule
        and its optimizers."""
        pp.sprint(f"{pp.COLOR.NORMAL}ENDING TASK {self.current_task}, loop {self.current_loop}")
        if callable(getattr(self.trainer.lightning_module, "on_task_end", None)):
            self.trainer.lightning_module.on_task_end()
        self._try_export_model_checkpoint()
        self._next()
        torch.cuda.empty_cache()
        gc.collect()

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        self._try_save_trained_model()
        self._try_save_dreams()

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_task": self.current_task, "current_loop": self.current_loop}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_task = state_dict["current_task"]
        self.current_loop = state_dict['current_loop']

    def _reset_fitting(self) -> None:
        # removed as for fast_dev_run it sets _last_val_dl_reload_epoch as 0 and not as -inf, causing 
        # to not reload dataloaders in function _reload_evaluation_dataloaders() 
        #self.trainer.reset_val_dataloader() 

        #TODO may be not needed
        #self.trainer.reset_train_dataloader()
        self.trainer.state.fn = TrainerFn.FITTING
        self.trainer.training = True

    def _reset_testing(self) -> None:
        self.trainer.reset_test_dataloader()
        self.trainer.state.fn = TrainerFn.TESTING
        self.trainer.testing = True

    def __getattr__(self, key) -> Any:
        # requires to be overridden as attributes of the wrapped loop are being accessed
        if key not in self.__dict__:
            return getattr(self.fit_loop, key)
        return self.__dict__[key]

    def _try_save_model_layer_stats(self):
        if(self.model_stats is not None and self.cfg_save.layer_stats is not None and not self.fast_dev_run):
            stripped_dict = {}
            for k, v in self.model_stats.items():
                stripped_dict[k] = v.get_const_data()
            filepath, name = self._generate_save_path("stat")
            torch.save(stripped_dict, f=filepath / name) 
    
    def _try_load_model_layer_stats(self, strict=True):
        if(self.cfg_load.layer_stats):
            if(self.model_stats is None):
                self.model_stats = ModelLayerStatistics(
                    model=self.trainer.lightning_module,
                    device=self.cfg_layer_stats.device,
                    hook_verbose=self.cfg_layer_stats.verbose,
                    flush_to_disk=self.cfg_layer_stats.flush_to_disk,
                    hook_to=self.cfg_layer_stats.hook_to,
                )
                self.model_stats.unhook()
            path = self._generate_load_path('layer_stats')
            loaded = torch.load(path)
            self.model_stats.set_layer_stats_from(loaded)
            self.model_stats = self.model_stats.get_stats()
            pp.sprint(f'{pp.COLOR.NORMAL}INFO: Loaded layer stats from {path}')
            
