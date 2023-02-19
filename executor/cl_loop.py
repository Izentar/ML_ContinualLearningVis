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
        reload_model_after_loop: bool = False,
        reinit_model_after_loop: bool = False,
        export_path: Optional[str] = None,
        enable_dreams=True,
        fast_dev_run_epochs=None,
        fast_dev_run=False,
        data_passer=None,
        num_loops=None,
        run_without_training=False,
        early_finish_at=-1,
        swap_datasets=False,
        weight_reset_sanity_check=False,
        enable_checkpoint=False,
        save_trained_model:str=None,
        load_model:str=None,
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
        self.export_path = Path(export_path) if export_path is not None else Path('./model_save/')
        Path.mkdir(self.export_path, parents=True, exist_ok=True, mode=770)

        
        self.reload_model_after_loop = reload_model_after_loop
        self.reinit_model_after_loop = reinit_model_after_loop
        self.enable_dreams = enable_dreams
        self.fast_dev_run_epochs = fast_dev_run_epochs
        self.fast_dev_run = fast_dev_run
        self.enable_data_parser = data_passer is not None
        self.data_passer = data_passer if data_passer is not None else {}
        self.run_without_training = run_without_training
        self.custom_advance_f = None
        self.early_finish_at = early_finish_at
        self.swap_datasets = swap_datasets
        self.weight_reset_sanity_check = weight_reset_sanity_check
        self.enable_checkpoint = enable_checkpoint
        self.save_trained_model = save_trained_model
        self.load_model = load_model

        if(self.swap_datasets and (self.num_tasks != 1 or self.num_loops % 2 == 1 or self.reload_model_after_loop == False)):
            raise Exception(f'Wrong variables set for "swap_datasets" flag. \
--num_tasks:"{self.num_tasks}" --num_loops:"{self.num_loops}" --reload_model_after_loop:"{self.reload_model_after_loop}"\n\
Values must be --num_tasks:"1" --num_loops:"%2" --reload_model_after_loop:"True"')

        if(self.reinit_model_after_loop and self.reload_model_after_loop):
            raise Exception("ERROR: reinit_model_after_loop and reload_model_after_loop cannot be both true")

        if(self.swap_datasets):
            print(f"INFO: CLLoop in swap_datasets mode.")

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
        if(self.enable_data_parser):
            self.data_passer['current_task'] = self.current_task
            self.data_passer['current_loop'] = self.current_loop
            self.data_passer['model_train_end_f'] = None
            if (self.early_finish_at >= 0 and self.data_passer['current_loop'] >= self.early_finish_at):
                self.data_passer['model_train_end_f'] = -1
        else:
            self.data_passer['current_task'] = -1
            self.data_passer['current_loop'] = -1
            self.data_passer['model_train_end_f'] = None
        self.data_passer['epoch_per_task'] = self.epochs_per_task[self.current_task]

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_tasks` from the `BaseCLDataModule` instance and store the
        original weights of the model."""
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)
        self._try_load_model()
        self.trainer.datamodule.setup_tasks()
        if(self.reload_model_after_loop):
            # need deepcopy, because state_dict reference the tensor, not its copy
            self.lightning_module_state_dict = deepcopy(self.trainer.lightning_module.state_dict())
            if(self.weight_reset_sanity_check):
                self.state_dict_sanity_check_val = self.trainer.lightning_module.model.get_objective_layer().weight.cpu()  
        if(self.reinit_model_after_loop):
            if(self.weight_reset_sanity_check):
                self.state_dict_sanity_check_val = self.trainer.lightning_module.model.get_objective_layer().weight.cpu() 
        self._update_data_passer()

    def _try_generate_dream(self):
        if (self.current_loop > 0 and self.enable_dreams):
            print(f"DREAMING DURING TASK: {self.current_task}, loop {self.current_loop}")
            #self.trainer.datamodule.setup_task_index(self.current_task)
            self.trainer.datamodule.generate_synthetic_data(
                self.trainer.lightning_module, self.current_task
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
        if(self.reload_model_after_loop and self.current_loop > 0):
            self.trainer.lightning_module.load_state_dict(
                self.lightning_module_state_dict, strict=True
            )
            self.trainer.strategy.setup_optimizers(self.trainer)
            self._model_weigth_sanity_check()
        if(self.reinit_model_after_loop and self.current_loop > 0):
            self.trainer.lightning_module.init_weights()
            self._model_weigth_sanity_check()

    def _setup_loop(self):
        # Set the max number of epochs for this task
        max_epochs = self.fast_dev_run_epochs if self.fast_dev_run and self.fast_dev_run_epochs is not None else self.epochs_per_task[self.current_task]
        self.replace(
            fit_loop=FitLoop(max_epochs=max_epochs)
        )
        self.custom_advance_f = self.fit_loop.run if not self.run_without_training else lambda: print("INFO: SKIPPING TRAINING")

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_task_index` from the `BaseCLDataModule` instance."""
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)

        if(self.swap_datasets):
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
        self.current_task += 1 if self.current_task < self.num_tasks - 1 else 0
        self.current_loop += 1
        self._update_data_passer()

    def _try_export_model_checkpoint(self):
        if self.enable_checkpoint:
            folder = datetime.datetime.now().strftime("%d-%m-%Y")
            time = datetime.datetime.now().strftime("%H:%M:%S")
            self.trainer.save_checkpoint(
                self.export_path / f"{folder}" / f"checkpoint.{type(self.trainer.lightning_module.model).__name__}.loop_{self.current_loop}.{time}.pt"
            )

    def _try_save_trained_model(self):
        if(self.save_trained_model is not None):
            folder = datetime.datetime.now().strftime("%d-%m-%Y")
            time = datetime.datetime.now().strftime("%H:%M:%S")
            self.trainer.save_checkpoint(
                self.export_path / f"{folder}" / f"trained.{type(self.trainer.lightning_module.model).__name__}.loop_{self.current_loop}.{time}.pt",
                weights_only=True
            )

    def _try_load_model(self):
        if(self.load_model is not None):
            find_path = self.export_path / self.load_model
            path = glob.glob(str(find_path), recursive=True)
            if(len(path) != 1):
                raise Exception(f'Cannot load model - no or too many matching filenames. From "{find_path}" found only these paths: {path}')
            path = path[0]
            if('checkpoint' in self.load_model):
                self.trainer.lightning_module.load_from_checkpoint(path)
                print(f'INFO: Loaded model "{path}"')
            elif('trained' in self.load_model):
                checkpoint = torch.load(path)
                self.trainer.lightning_module.load_state_dict(checkpoint["state_dict"])
                print(f'INFO: Loaded model "{path}"')
            else:
                raise Exception(f'Unknown filename to load model. File: {self.load_model}')

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.custom_advance_f()

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
