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
from colorama import Fore, Back, Style
from collections.abc import Sequence
from loss_function import image_regularization

from dataclasses import dataclass, field
from argparse import Namespace

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
        use_input_img_var_reg_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
        bn_reg_scale: float = 1e2
        use_var_img_reg_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
        var_scale: float = 2.5e-5
        use_l2_img_reg_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
        l2_coeff: float = 1e-05

    @dataclass
    class Model():
        reload_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False
        reinit_at: Union[str, bool, int, list[str], list[bool], list[int], None] = False

    @dataclass
    class Save():
        enable_checkpoint: bool = False
        model_inner_path: str = None
        model: bool = False
        dreams: str = None
        layer_stats: str = None
        export_path: Optional[str] = None

        def __post_init__(self):
            self.model_inner_path = Path(self.model_inner_path) if self.model_inner_path is not None else Path("")
            if self.export_path is not None:
                self.export_path = Path(self.export_path)
            else:
                self.export_path = Path(default_export_path)

    @dataclass
    class Load():
        model: str = None
        dreams: str = None
        layer_stats: str = None
        export_path: Optional[str] = None

        def __post_init__(self):
            self.model = Path(self.model) if self.model is not None else self.model
            self.dreams = Path(self.dreams) if self.dreams is not None else self.dreams
            if self.export_path is not None:
                self.export_path = Path(self.export_path)
            else:
                self.export_path = Path(default_export_path)
        
    @dataclass
    class LayerStats():
        use_at: Union[str, bool, int, list[str], list[bool], list[int], None] = None
        hook_to: Union[list[str], None] = False
        device: str = 'cuda'
        flush_to_disk: bool = False
        verbose: bool = False

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
            device: str = 'cuda'

    CONFIG_MAP = {
        'vis': Visualization,
        'model': Model,
        'save': Save,
        'load': Load,
        'mean_norm': LayerLoss.MeanNorm,
        'grad_pruning': LayerLoss.GradPruning,
        'grad_activ_pruning': LayerLoss.GradActivePruning,
        'layer_stats': LayerStats,
        'cfg': Config,
    }

    def __init__(
        self,
        plan: list[list[int]],
        cfg_map: dict = None,
        args = None,
        fast_dev_run_epochs=None,
        fast_dev_run=False,
        data_passer=None,
        weight_reset_sanity_check=False,
        data_module=None,
        progress_bar=None,
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

        self._map_cfg(args=args, cfg_map=cfg_map, plan=plan)

        self.current_task: int = 0
        self.current_loop: int = 0
        self.previous_task:int = 0
        self.previous_loop:int = 0
        
        self.fast_dev_run_epochs = fast_dev_run_epochs
        self.fast_dev_run = fast_dev_run
        self.weight_reset_sanity_check = weight_reset_sanity_check
        self.data_module = data_module
        self.progress_bar = progress_bar

        self.enable_data_parser = data_passer is not None
        self.data_passer = data_passer if data_passer is not None else {}
        self.custom_advance_f = None
        self.model_stats = None

        if utils.check_python_enabled(self.cfg_model.reinit_at) and utils.check_python_enabled(self.cfg_model.reload_at):
            raise Exception("ERROR: cfg_model.reinit_at and cfg_model.reload_at cannot be both true")

        for l in self.cfg.plan:
            if((not isinstance(l, Sequence)) or len(l) < 1):
                raise Exception(f'Bad plan: {self.cfg.plan}')

    def _map_cfg(self, args, cfg_map:dict, plan):
        if(args is None and cfg_map is None):
            raise Exception(f"No config provided.")
        if(cfg_map is not None):
            for k in cfg_map.keys():
                if(k not in self.CONFIG_MAP):
                    raise Exception(f"Unknown config map key: {k}")

        if(args is not None):
            not_from = Namespace
            cfg = utils.get_obj_dict(args.loop, not_from)
            if('plan' in cfg):
                self.cfg = CLLoop.Config(
                    **cfg
                )
            else:
                self.cfg = CLLoop.Config(
                    plan=plan, **cfg
                )
            self.cfg_vis=CLLoop.Visualization(
                **utils.get_obj_dict(args.loop.vis, not_from)
            )
            self.cfg_model=CLLoop.Model(
                **utils.get_obj_dict(args.loop.model, not_from)
            )
            self.cfg_save=CLLoop.Save(
                **utils.get_obj_dict(args.loop.save, not_from)
            )
            self.cfg_load=CLLoop.Load(
                **utils.get_obj_dict(args.loop.load, not_from)
            )
            self.cfg_mean_norm=CLLoop.LayerLoss.MeanNorm(
                **utils.get_obj_dict(args.loop.layerloss.mean_norm, not_from)
            )
            self.cfg_grad_pruning=CLLoop.LayerLoss.GradPruning(
                **utils.get_obj_dict(args.loop.layerloss.grad_pruning, not_from)
            )
            self.cfg_grad_activ_pruning=CLLoop.LayerLoss.GradActivePruning(
                **utils.get_obj_dict(args.loop.layerloss.grad_activ_pruning, not_from)
            )
            self.cfg_layer_stats=CLLoop.LayerStats(
                **utils.get_obj_dict(args.loop.layer_stats, not_from)
            )

        if(cfg_map is not None):
            for k, cfg_class in self.CONFIG_MAP.items():
                name = f'cfg_{k}' if 'cfg' not in k else 'cfg'
                if(k not in cfg_map):
                    if(args is None): # setup if not set
                        setattr(self, name, cfg_class())
                else: # override
                    setattr(self, name, cfg_map[k])

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
        print(f"HOOKING TO MODEL - GATHER STATS: task: {self.current_task}, loop {self.current_loop}")

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

    def _try_generate_dream(self):
        if utils.check_python_index(self.cfg_vis.generate_at, self.cfg.num_loops, self.current_loop):
            layer_hook_obj = []
            layer_handles = []
            input_image_train_after_obj = []

            if(utils.check_python_index(self.cfg_mean_norm.use_at, self.cfg.num_loops, self.current_loop)):
                print(f"HOOKING TO MODEL - LOSS FUNCTION: task: {self.current_task}, loop {self.current_loop}")
                self._try_load_model_layer_stats()
                tmp = layerloss.MeanNorm(device=self.cfg_mean_norm.device, del_cov_after=self.cfg_mean_norm.del_cov_after, scaling=self.cfg_mean_norm.scaling)
                layer_hook_obj.append(tmp)
                layer_handles.append(layer_stat_framework.hook_model_stats(
                    model=self.trainer.lightning_module, 
                    stats=self.model_stats, 
                    fun=tmp.hook_fun, 
                    hook_to=self.cfg_mean_norm.hook_to
                ))
            if(utils.check_python_index(self.cfg_grad_pruning.use_at, self.cfg.num_loops, self.current_loop)):
                print(f"HOOKING TO MODEL - GRAD PRUNING FUNCTION: task: {self.current_task}, loop {self.current_loop}")
                self._try_load_model_layer_stats()
                tmp = layerloss.LayerGradPruning(device=self.cfg_grad_pruning.device, percent=self.cfg_grad_pruning.percent)
                layer_hook_obj.append(tmp)
                layer_handles.append(layer_stat_framework.hook_model_stats(
                    model=self.trainer.lightning_module, 
                    stats=self.model_stats, 
                    fun=tmp.hook_fun, 
                    hook_to=self.cfg_grad_pruning.hook_to
                ))
            if(utils.check_python_index(self.cfg_grad_activ_pruning.use_at, self.cfg.num_loops, self.current_loop)):
                print(f"HOOKING TO MODEL - GRAD PRUNING FUNCTION: task: {self.current_task}, loop {self.current_loop}")
                self._try_load_model_layer_stats()
                tmp = layerloss.LayerGradActivationPruning(device=self.cfg_grad_activ_pruning.device, percent=self.cfg_grad_activ_pruning.percent)
                layer_hook_obj.append(tmp)
                layer_handles.append(utils.hook_model(
                    model=self.trainer.lightning_module, 
                    fun=tmp.hook_fun, 
                    hook_to=self.cfg_grad_activ_pruning.hook_to
                ))

            if(utils.check_python_index(self.cfg_vis.use_input_img_var_reg_at, self.cfg.num_loops, self.current_loop)):
                tmp = layerloss.DeepInversionFeatureLoss(bn_reg_scale=self.cfg_vis.bn_reg_scale)
                input_image_train_after_obj.append(tmp)
                layer_handles.append(utils.hook_model(
                    model=self.trainer.lightning_module, 
                    fun=tmp.hook_fun, 
                    hook_to=['BatchNorm2d'] # only works for BatchNorm2d
                ))

            if(utils.check_python_index(self.cfg_vis.use_var_img_reg_at, self.cfg.num_loops, self.current_loop)):
                tmp = image_regularization.VariationRegularization(scale=self.cfg_vis.var_scale)
                input_image_train_after_obj.append(tmp)

            if(utils.check_python_index(self.cfg_vis.use_l2_img_reg_at, self.cfg.num_loops, self.current_loop)):
                tmp = image_regularization.L2Regularization(coefficient=self.cfg_vis.l2_coeff)
                input_image_train_after_obj.append(tmp)
            
            print(f"DREAMING DURING TASK: {self.current_task}, loop {self.current_loop}")
            #self.trainer.datamodule.setup_task_index(self.current_task)
            self.trainer.datamodule.generate_synthetic_data(
                model=self.trainer.lightning_module, 
                task_index=self.current_task, 
                layer_hook_obj=layer_hook_obj,
                input_image_train_after_obj=input_image_train_after_obj,
            )
            if(len(layer_handles) != 0):
                for l in layer_handles:
                    unhook(l)
            print("DREAMING END")

    def _model_weigth_sanity_check(self):
        if(self.weight_reset_sanity_check):
            test_sanity_val = self.trainer.lightning_module.model.get_objective_layer().weight.cpu()
            if torch.allclose(self.state_dict_sanity_check_val, test_sanity_val):
                print("Model reset success")
                return
            print('FAIL: Model reset fail sanity check.')
            print(self.state_dict_sanity_check_val)
            print(test_sanity_val)

    def _try_reset_model(self):
        # restore the original weights + optimizers and schedulers.
        if utils.check_python_index(self.cfg_model.reload_at, self.cfg.num_loops, self.current_loop):
            print('INFO: Model reloaded.')
            self.trainer.lightning_module.load_state_dict(
                self.lightning_module_state_dict, strict=True
            )
            self.trainer.strategy.setup_optimizers(self.trainer)
            self._model_weigth_sanity_check()
        if utils.check_python_index(self.cfg_model.reinit_at, self.cfg.num_loops, self.current_loop):
            print('INFO: Model reinit.')
            self.trainer.lightning_module.init_weights()
            self._model_weigth_sanity_check()

    @property
    def epoch_num(self):
        if(self.fast_dev_run and self.fast_dev_run_epochs is not None):
            return self.fast_dev_run_epochs
        if(len(self.cfg.plan[self.current_task]) <= self.current_loop):
            tmp = self.cfg.plan[self.current_task][-1]
            print(f'WARNING: At loop {self.current_loop} selected last epoch per task "{tmp}" because list index out of range.')
            return tmp
        return self.cfg.plan[self.current_task][self.current_loop]

    def _setup_loop(self):
        # Set the max number of epochs for this task
        max_epochs = self.epoch_num
        self.replace(
            fit_loop=FitLoop(max_epochs=max_epochs)
        )
        if(utils.check_python_index(self.cfg_layer_stats.use_at, self.cfg.num_loops, self.current_loop)):
            print("INFO: HOOKING UP LOOP TO GATHER STATISTICS")
            self.custom_advance_f = self._gather_model_layer_stats_advance_f # run custon data gathering loop
        elif(utils.check_python_index(self.cfg.train_at, self.cfg.num_loops, self.current_loop)):
            print("INFO: HOOKING UP NORMAL LOOP")
            self.custom_advance_f = self.fit_loop.run # run subloop - FitLoop
        else:
            self.custom_advance_f = lambda: print(f"{Fore.RED}INFO: SKIPPING ANY TRAINING at loop {self.current_loop}{Style.RESET_ALL}")

    def _export_path_contains(self, other, export_path):
        size = len(export_path.parents) + 1
        if(export_path == other.parents[- size]):
            return True
        return False

    def _gen_folder_name_by_time(self, export_path, dtype:str=None):
        folder = datetime.datetime.now().strftime("%d-%m-%Y")
        time = datetime.datetime.now().strftime("%H-%M-%S")
        if(dtype is None):
            dtype = ""
        if(self._export_path_contains(self.cfg_save.model_inner_path, export_path=export_path)):
            ret = self.cfg_save.model_inner_path / dtype / folder
        else:
            ret = export_path / self.cfg_save.model_inner_path / dtype / folder
        ret.mkdir(parents=True, exist_ok=True)
        return ret, time

    def _try_save_dreams(self):
        if(self.cfg_save.dreams is not None and not self.fast_dev_run):
            filepath, time = self._gen_folder_name_by_time(export_path=self.cfg_save.export_path, dtype=self.cfg_save.dreams)
            filename = f"dreams.loop_{self.current_loop}.{type(self.trainer.lightning_module.model).__name__}.{type(self.trainer.lightning_module).__name__}.{time}.pt"
            Path.mkdir(filepath, parents=True, exist_ok=True)
            self.trainer.datamodule.save_dream_dataset(filepath / filename)

    def _try_load_dreams(self):
        if(self.cfg_load.dreams is not None):
            if(self._export_path_contains(self.cfg_load.dreams, self.cfg_load.export_path)):
                find_path = self.cfg_load.dreams
            else:
                find_path = self.cfg_load.export_path / self.cfg_load.dreams
            path = glob.glob(str(find_path), recursive=True)
            if('dreams.' not in self.cfg_load.dreams):
                raise Exception(f'Unknown filename to load dreams. May be the wrong filetype. File: {find_path}')
            if(len(path) != 1):
                raise Exception(f'Cannot load dreams - no or too many matching filenames. From "{find_path}" found only these paths: {path}')
            path = path[0]
            self.trainer.datamodule.load_dream_dataset(path)

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_task_index` from the `BaseCLDataModule` instance."""
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)

        if(utils.check_python_index(self.cfg_vis.clear_dataset_at, self.cfg.num_loops, self.current_loop)):
            print(f'INFO: Dreams cleared at loop {self.current_loop}, task {self.current_task}')
            self.trainer.datamodule.clear_dreams_dataset()
        self._try_generate_dream()
        self._update_data_passer()
        print(f"STARTING TASK {self.current_task}, loop {self.current_loop} -- classes in task {self.trainer.datamodule.get_task_classes(self.current_task)}")
        
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
            filepath, time = self._gen_folder_name_by_time(export_path=self.cfg_save.export_path)
            self.trainer.save_checkpoint(
                filepath / f"checkpoint.loop_{self.current_loop}.{type(self.trainer.lightning_module.model).__name__}.{type(self.trainer.lightning_module).__name__}.{time}.pt"
            )

    def _try_save_trained_model(self):
        if(self.cfg_save.model and not self.fast_dev_run):
            filepath, time = self._gen_folder_name_by_time(export_path=self.cfg_save.export_path)
            self.trainer.save_checkpoint(
                filepath / f"trained.loop_{self.current_loop}.{type(self.trainer.lightning_module.model).__name__}.{type(self.trainer.lightning_module).__name__}.{time}.pt",
                weights_only=True
            )

    def _try_load_model(self):
        if(self.cfg_load.model is not None):
            if(self._export_path_contains(self.cfg_load.model, self.cfg_load.export_path)):
                find_path = self.cfg_load.model
            else:
                find_path = self.cfg_load.export_path / self.cfg_load.model
            path = glob.glob(str(find_path), recursive=True)
            if(len(path) != 1):
                raise Exception(f'Cannot load model - no or too many matching filenames. From "{find_path}" found only these paths: {path}')
            path = Path(path[0])
            if('checkpoint.' in path.name):
                pass
            elif('trained.' in path.name):
                checkpoint = torch.load(path)
                self.trainer.lightning_module.load_state_dict(checkpoint["state_dict"])
                print(f'INFO: Loaded model "{path}"')
            else:
                raise Exception(f'Unknown filename to load model. May be the wrong filetype. File: {find_path}')

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.custom_advance_f(*args, **kwargs)

    def on_advance_end(self) -> None:
        """Used to save the weights of the current task and reset the LightningModule
        and its optimizers."""
        print(f"ENDING TASK {self.current_task}, loop {self.current_loop}")
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
            Path(self.cfg_save.layer_stats).parent.mkdir(parents=True, exist_ok=True)
            filepath, time = self._gen_folder_name_by_time(export_path=self.cfg_save.export_path)
            torch.save(stripped_dict, f=filepath / f'layer_stats.loop_{self.current_loop}.{self.cfg_save.layer_stats}.{time}.ls') 
    
    def _try_load_model_layer_stats(self, strict=True):
        if(self.cfg_load.layer_stats is not None):
            if(self.model_stats is None):
                self.model_stats = ModelLayerStatistics(
                    model=self.trainer.lightning_module,
                    device=self.cfg_layer_stats.device,
                    hook_verbose=self.cfg_layer_stats.verbose,
                    flush_to_disk=self.cfg_layer_stats.flush_to_disk,
                    hook_to=self.cfg_layer_stats.hook_to,
                )
                self.model_stats.unhook()
            loaded = torch.load(self.cfg_load.layer_stats)
            self.model_stats.set_layer_stats_from(loaded)
            self.model_stats = self.model_stats.get_stats()
            
