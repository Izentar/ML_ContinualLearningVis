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
        reset_model_after_task: bool = False,
        export_path: Optional[str] = None,
        enable_dreams=True,
        fast_dev_run_epochs=None,
        fast_dev_run=False,
        data_passer=None,
        num_loops=None,
        run_without_training=False,
        dream_only_once=False,
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
        self.current_task_loop: int = 0
        self.export_path = export_path
        if self.export_path is not None:
            self.export_path = Path(export_path)
        self.reset_model_after_task = reset_model_after_task
        self.enable_dreams = enable_dreams
        self.fast_dev_run_epochs = fast_dev_run_epochs
        self.fast_dev_run = fast_dev_run
        self.enable_data_parser = data_passer is not None
        self.data_passer = data_passer if data_passer is not None else {}
        self.run_without_training = run_without_training
        self.custom_advance_f = None
        self.dream_only_once = dream_only_once

    @property
    def done(self) -> bool:
        return self.current_task_loop >= self.num_loops

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""
        self.current_task = 0
        self.current_task_loop = 0
        self.fit_loop.reset()
        #if self.trainer.fast_dev_run:
        #    self.num_tasks = fast_dev_run_config["num_tasks"]

    def _update_data_passer(self):
        if(self.enable_data_parser):
            self.data_passer['current_task'] = self.current_task
            self.data_passer['current_task_loop'] = self.current_task_loop
            self.data_passer['model_train_end_f'] = lambda: None
            if (self.data_passer['current_task_loop'] >= 1 and self.dream_only_once):
                self.data_passer['model_train_end_f'] = lambda: -1
        else:
            self.data_passer['current_task'] = -1
            self.data_passer['current_task_loop'] = -1
            self.data_passer['model_train_end_f'] = lambda: None

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_tasks` from the `BaseCLDataModule` instance and store the
        original weights of the model."""
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)
        self.trainer.datamodule.setup_tasks()
        if self.reset_model_after_task:
            self.lightning_module_state_dict = deepcopy(
                self.trainer.lightning_module.state_dict()
            )
        self._update_data_passer()

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_task_index` from the `BaseCLDataModule` instance."""
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)
        if (self.current_task_loop > 0 and self.enable_dreams):
            print(f"DREAMING FOR TASK: {self.current_task}, loop {self.current_task_loop}")
            #self.trainer.datamodule.setup_task_index(self.current_task)
            self.trainer.datamodule.generate_synthetic_data(
                self.trainer.lightning_module, self.current_task
            )
            
        self._update_data_passer()
        print(f"STARTING TASK {self.current_task}, loop {self.current_task_loop} -- classes {self.trainer.datamodule.get_task_classes(self.current_task)}")
        # restore the original weights + optimizers and schedulers.
        if self.reset_model_after_task:
            self.trainer.lightning_module.load_state_dict(
                self.lightning_module_state_dict
            )
            self.trainer.strategy.setup_optimizers(self.trainer)
        self.trainer.datamodule.setup_task_index(self.current_task, self.current_task_loop)
        # Set the max number of epochs for this task

        max_epochs = self.fast_dev_run_epochs if self.fast_dev_run and self.fast_dev_run_epochs is not None else self.epochs_per_task[self.current_task]
        self.replace(
            fit_loop=FitLoop(max_epochs=max_epochs)
        )
        self.custom_advance_f = self.fit_loop.run if not self.run_without_training else lambda: print("SKIPPING TRAINING")
        # TODO I think there is no function named like that
        if callable(getattr(self.trainer.lightning_module, "on_task_start", None)):
            self.trainer.lightning_module.on_task_start()

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.custom_advance_f()
        
        self.current_task += 1 if self.current_task < self.num_tasks - 1 else 0
        self.current_task_loop += 1
        self._update_data_passer()

    def on_advance_end(self) -> None:
        """Used to save the weights of the current task and reset the LightningModule
        and its optimizers."""
        print(f"ENDING TASK {self.current_task}, loop {self.current_task_loop}")
        if callable(getattr(self.trainer.lightning_module, "on_task_end", None)):
            self.trainer.lightning_module.on_task_end()
        if self.export_path is not None:
            self.trainer.save_checkpoint(
                self.export_path / f"model.{self.current_task_loop}.pt"
            )

        torch.cuda.empty_cache()
        gc.collect()

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""
        pass

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_task": self.current_task, "current_task_loop": self.current_task_loop}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_task = state_dict["current_task"]
        self.current_task_loop = state_dict['current_task_loop']

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
