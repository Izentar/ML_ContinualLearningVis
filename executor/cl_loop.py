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
from typing import Any, Dict, List, Optional

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
from pytorch_lightning.trainer.supporters import CombinedLoader
from utils import utils
from model.statistics.base import ModelLayerStatistics
from colorama import Fore, Back, Style

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
    def __init__(
        self,
        epochs_per_task: List[int],
        reload_model_at: bool = False,
        reinit_model_at: bool = False,
        export_path: Optional[str] = None,
        enable_dreams_gen_at:int=None,
        fast_dev_run_epochs=None,
        fast_dev_run=False,
        data_passer=None,
        num_loops=None,
        run_training_at=False,
        early_finish_at=-1,
        swap_datasets=False,
        weight_reset_sanity_check=False,
        enable_checkpoint:bool=False,
        save_trained_model:bool=False,
        save_model_inner_path:str=None,
        load_model:str=None,
        save_dreams:str=None,
        load_dreams:str=None,
        gather_layer_loss_at:int=None,
        use_layer_loss_at:int=None,
        layer_dataloader=None,
        data_module=None,
        progress_bar=None,
        layer_stats_hook_to:list[str]|None=False,
        layer_stats_verbose=False,
        layer_stats_flush_to_disk=False,
        layer_stats_loss_device='cuda:0',
        layer_stats_collect_device='cuda:0',
        advance_clear_dreams=False,
        layer_loss_del_cov_after=False,
        save_layer_stats=None,
        load_layer_stats=None,
    ) -> None:
        """
            epochs_per_task: list of epoches per task
            export_path: save model parameters to path on current task

            num_loops - if None then the same as num_tasks. It is used to loop over the num_tasks.
                If num_loops == num_tasks then nothing is changed. 
                It can be used to loop more than the num_tasks after 
                all tasks are done.
        """
        super().__init__()
        self.num_tasks = len(epochs_per_task)
        self.num_loops = self.num_tasks if num_loops is None else num_loops
        self.epochs_per_task = epochs_per_task
        self.current_task: int = 0
        self.current_loop: int = 0
        self.previous_task:int = 0
        self.previous_loop:int = 0
        self.export_path = Path(export_path) if export_path is not None else Path(default_export_path)
        self.export_path.mkdir(parents=True, exist_ok=True)
        self.save_model_inner_path = save_model_inner_path if save_model_inner_path is not None else ""
        
        self.reload_model_at = reload_model_at
        self.reinit_model_at = reinit_model_at
        self.enable_dreams_gen_at = enable_dreams_gen_at
        self.fast_dev_run_epochs = fast_dev_run_epochs
        self.fast_dev_run = fast_dev_run
        self.run_training_at = run_training_at
        self.early_finish_at = early_finish_at
        self.swap_datasets = swap_datasets
        self.weight_reset_sanity_check = weight_reset_sanity_check
        self.enable_checkpoint = enable_checkpoint
        self.save_trained_model = save_trained_model
        self.load_model = load_model
        self.save_dreams = save_dreams
        self.load_dreams = load_dreams
        self.gather_layer_loss_at = gather_layer_loss_at
        self.layer_dataloader = layer_dataloader
        self.use_layer_loss_at = use_layer_loss_at
        self.data_module = data_module
        self.progress_bar = progress_bar
        self.advance_clear_dreams = advance_clear_dreams

        self.enable_data_parser = data_passer is not None
        self.data_passer = data_passer if data_passer is not None else {}
        self.custom_advance_f = None
        self.model_stats = None
        self.layer_loss_del_cov_after = layer_loss_del_cov_after

        if(self.swap_datasets and (self.num_tasks != 1 or self.num_loops % 2 == 1 or self.reload_model_at == False)):
            raise Exception(f'Wrong variables set for "swap_datasets" flag. \
--num_tasks:"{self.num_tasks}" --num_loops:"{self.num_loops}" --reload_model_at:"{self.reload_model_at}"\n\
Values must be --num_tasks:"1" --num_loops:"%2" --reload_model_at:"True"')

        if utils.check_python_enabled(self.reinit_model_at) and utils.check_python_enabled(self.reload_model_at):
            raise Exception("ERROR: reinit_model_at and reload_model_at cannot be both true")

        if(self.swap_datasets):
            print(f"INFO: CLLoop in swap_datasets mode.")

        self.layer_stats_hook_to = layer_stats_hook_to
        self.layer_stats_verbose = layer_stats_verbose
        self.layer_stats_flush_to_disk = layer_stats_flush_to_disk
        self.layer_stats_loss_device = layer_stats_loss_device
        self.layer_stats_collect_device = layer_stats_collect_device
        self.load_layer_stats = load_layer_stats
        self.save_layer_stats = save_layer_stats

    @property
    def done(self) -> bool:
        return self.current_loop >= self.num_loops

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""
        self.current_task = 0
        self.current_loop = 0
        self.fit_loop.reset()
        #if self.trainer.fast_dev_run:
        #    self.num_tasks = fast_dev_run_config["num_tasks"]

    def _update_data_passer(self):
        self.data_passer['current_task'] = self.current_task
        self.data_passer['current_loop'] = self.current_loop
        self.data_passer['num_tasks'] = self.num_tasks
        self.data_passer['num_loops'] = self.num_loops
        if(self.enable_data_parser):
            self.data_passer['model_train_end_f'] = None
            if (self.early_finish_at >= 0 and self.data_passer['current_loop'] >= self.early_finish_at):
                self.data_passer['model_train_end_f'] = -1
        else:
            self.data_passer['model_train_end_f'] = None
        self.data_passer['epoch_per_task'] = self.epochs_per_task[self.current_task]

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_tasks` from the `BaseCLDataModule` instance and store the
        original weights of the model."""
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)
        self._try_load_model()
        self.trainer.datamodule.setup_tasks()
        if(utils.check_python_enabled(self.reload_model_at)):
            # need deepcopy, because state_dict reference the tensor, not its copy
            self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())
            if(self.weight_reset_sanity_check):
                self.state_dict_sanity_check_val = self.trainer.lightning_module.model.get_objective_layer().weight.cpu()  
        if(utils.check_python_enabled(self.reinit_model_at)):
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
            device=self.layer_stats_collect_device,
            hook_verbose=self.layer_stats_verbose,
            progress_bar=self.progress_bar,
            flush_to_disk=self.layer_stats_flush_to_disk,
            hook_to=self.layer_stats_hook_to,
            fast_dev_run=self.fast_dev_run,
        )
        self._try_save_model_layer_stats()

    def _try_generate_dream(self):
        main_enable = utils.check_python_index(self.enable_dreams_gen_at, self.num_loops, self.current_loop)
        if main_enable:
            layer_loss = None
            if(utils.check_python_index(self.use_layer_loss_at, self.num_loops, self.current_loop)):
                print(f"HOOKING TO MODEL - LOSS FUNCTION: task: {self.current_task}, loop {self.current_loop}")
                self._try_load_model_layer_stats()
                layer_loss = layer_stat_framework.LayerLoss(device=self.layer_stats_loss_device, del_cov_after=self.layer_loss_del_cov_after)
                layer_stat_framework.hook_model_stats(
                    model=self.trainer.lightning_module, 
                    stats=self.model_stats, 
                    fun=layer_loss.hook_fun, 
                    hook_to=self.layer_stats_hook_to
                )
            
            print(f"DREAMING DURING TASK: {self.current_task}, loop {self.current_loop}")
            #self.trainer.datamodule.setup_task_index(self.current_task)
            self.trainer.datamodule.generate_synthetic_data(
                model=self.trainer.lightning_module, 
                task_index=self.current_task, 
                layer_loss_obj=layer_loss,
            )

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
        if utils.check_python_index(self.reload_model_at, self.num_loops, self.current_loop):
            print('INFO: Model reloaded.')
            self.trainer.lightning_module.load_state_dict(
                self.lightning_module_state_dict, strict=True
            )
            self.trainer.strategy.setup_optimizers(self.trainer)
            self._model_weigth_sanity_check()
        if utils.check_python_index(self.reinit_model_at, self.num_loops, self.current_loop):
            print('INFO: Model reinit.')
            self.trainer.lightning_module.init_weights()
            self._model_weigth_sanity_check()

    def _setup_loop(self):
        # Set the max number of epochs for this task
        max_epochs = self.fast_dev_run_epochs if self.fast_dev_run and self.fast_dev_run_epochs is not None else self.epochs_per_task[self.current_task]
        self.replace(
            fit_loop=FitLoop(max_epochs=max_epochs)
        )
        if(utils.check_python_index(self.gather_layer_loss_at, self.num_loops, self.current_loop)):
            print("INFO: HOOKING UP LOOP TO GATHER STATISTICS")
            self.custom_advance_f = self._gather_model_layer_stats_advance_f # run custon data gathering loop
        elif(utils.check_python_index(self.run_training_at, self.num_loops, self.current_loop)):
            print("INFO: HOOKING UP NORMAL LOOP")
            self.custom_advance_f = self.fit_loop.run # run subloop - FitLoop
        else:
            self.custom_advance_f = lambda: print(f"{Fore.RED}INFO: SKIPPING ANY TRAINING at loop {self.current_loop}{Style.RESET_ALL}")

    def _gen_folder_name_by_time(self, dtype:str=None):
        folder = datetime.datetime.now().strftime("%d-%m-%Y")
        time = datetime.datetime.now().strftime("%H-%M-%S")
        if(dtype is None):
            dtype = ""
        ret = self.export_path / f"{self.save_model_inner_path}" / dtype / folder
        ret.mkdir(parents=True, exist_ok=True)
        return ret, time

    def _try_save_dreams(self):
        if(self.save_dreams is not None and not self.fast_dev_run):
            filepath, time = self._gen_folder_name_by_time(self.save_dreams)
            filename = f"dreams.loop_{self.current_loop}.{type(self.trainer.lightning_module.model).__name__}.{type(self.trainer.lightning_module).__name__}.{time}.pt"
            Path.mkdir(filepath, parents=True, exist_ok=True)
            self.trainer.datamodule.save_dream_dataset(filepath / filename)

    def _try_load_dreams(self):
        if(self.load_dreams is not None):
            find_path = self.export_path / self.load_dreams
            path = glob.glob(str(find_path), recursive=True)
            if('dreams.' not in self.load_dreams):
                raise Exception(f'Unknown filename to load dreams. May be the wrong filetype. File: {find_path}')
            if(len(path) != 1):
                raise Exception(f'Cannot load dreams - no or too many matching filenames. From "{find_path}" found only these paths: {path}')
            path = path[0]
            self.trainer.datamodule.load_dream_dataset(path)

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_task_index` from the `BaseCLDataModule` instance."""
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)

        if(self.advance_clear_dreams or self.swap_datasets):
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
        self.current_task += 1 if self.current_task < self.num_tasks - 1 else 0
        self.current_loop += 1
        self._update_data_passer()

    def _try_export_model_checkpoint(self):
        if self.enable_checkpoint and not self.fast_dev_run:
            filepath, time = self._gen_folder_name_by_time()
            self.trainer.save_checkpoint(
                filepath / f"checkpoint.loop_{self.current_loop}.{type(self.trainer.lightning_module.model).__name__}.{type(self.trainer.lightning_module).__name__}.{time}.pt"
            )

    def _try_save_trained_model(self):
        if(self.save_trained_model and not self.fast_dev_run):
            filepath, time = self._gen_folder_name_by_time()
            self.trainer.save_checkpoint(
                filepath / f"trained.loop_{self.current_loop}.{type(self.trainer.lightning_module.model).__name__}.{type(self.trainer.lightning_module).__name__}.{time}.pt",
                weights_only=True
            )

    def _try_load_model(self):
        if(self.load_model is not None):
            find_path = self.export_path / self.load_model
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
        if(self.model_stats is not None and self.save_layer_stats is not None and not self.fast_dev_run):
            stripped_dict = {}
            for k, v in self.model_stats.items():
                stripped_dict[k] = v.get_const_data()
            Path(self.save_layer_stats).parent.mkdir(parents=True, exist_ok=True)
            filepath, time = self._gen_folder_name_by_time()
            torch.save(stripped_dict, f=filepath / f'layer_stats.loop_{self.current_loop}.{self.save_layer_stats}.{time}.ls') 
    
    def _try_load_model_layer_stats(self, strict=True):
        if(self.load_layer_stats is not None):
            if(self.model_stats is None):
                self.model_stats = ModelLayerStatistics(
                    model=self.trainer.lightning_module,
                    device=self.layer_stats_collect_device,
                    hook_verbose=self.layer_stats_verbose,
                    flush_to_disk=self.layer_stats_flush_to_disk,
                    hook_to=self.layer_stats_hook_to,
                )
                self.model_stats.unhook()
            loaded = torch.load(self.load_layer_stats)
            self.model_stats.set_layer_stats_from(loaded)
            self.model_stats = self.model_stats.get_stats()
            
