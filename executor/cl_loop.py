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
    ) -> None:
        """
            epochs_per_task: list of epoches per task
            export_path: save model parameters to path on current task
        """
        super().__init__()
        self.num_tasks = len(epochs_per_task)
        self.epochs_per_task = epochs_per_task
        self.current_task: int = 0
        self.export_path = export_path
        if self.export_path is not None:
            self.export_path = Path(export_path)
        self.reset_model_after_task = reset_model_after_task

    @property
    def done(self) -> bool:
        return self.current_task >= self.num_tasks

    def connect(self, fit_loop: FitLoop) -> None:
        self.fit_loop = fit_loop

    def reset(self) -> None:
        """Nothing to reset in this loop."""
        self.current_task = 0
        self.fit_loop.reset()
        #if self.trainer.fast_dev_run:
        #    self.num_tasks = fast_dev_run_config["num_tasks"]

    def on_run_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_tasks` from the `BaseCLDataModule` instance and store the
        original weights of the model."""
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)
        self.trainer.datamodule.setup_tasks()
        if self.reset_model_after_task:
            self.lightning_module_state_dict = deepcopy(
                self.trainer.lightning_module.state_dict()
            )

    def on_advance_start(self, *args: Any, **kwargs: Any) -> None:
        """Used to call `setup_task_index` from the `BaseCLDataModule` instance."""
        print(f"STARTING TASK {self.current_task} -- classes {self.trainer.datamodule.get_task_classes(self.current_task)}")
        assert isinstance(self.trainer.datamodule, BaseCLDataModule)
        if (self.current_task > 0):
            self.trainer.datamodule.setup_task_index(self.current_task)
            self.trainer.datamodule.generate_synthetic_data(
                self.trainer.lightning_module, self.current_task
            )
        # restore the original weights + optimizers and schedulers.
        if self.reset_model_after_task:
            self.trainer.lightning_module.load_state_dict(
                self.lightning_module_state_dict
            )
            self.trainer.strategy.setup_optimizers(self.trainer)
        self.trainer.datamodule.setup_task_index(self.current_task)
        # Set the max number of epochs for this task
        self.replace(
            fit_loop=FitLoop(max_epochs=self.epochs_per_task[self.current_task])
        )
        # TODO I think there is no function named like that
        if callable(getattr(self.trainer.lightning_module, "on_task_start", None)):
            self.trainer.lightning_module.on_task_start()

    def advance(self, *args: Any, **kwargs: Any) -> None:
        """Used to the run a fitting and testing on the current hold."""
        self._reset_fitting()  # requires to reset the tracking stage.
        self.fit_loop.run()
        self.current_task += 1  # increment task tracking number.

    def on_advance_end(self) -> None:
        """Used to save the weights of the current task and reset the LightningModule
        and its optimizers."""
        if callable(getattr(self.trainer.lightning_module, "on_task_end", None)):
            self.trainer.lightning_module.on_task_end()
        if self.export_path is not None:
            self.trainer.save_checkpoint(
                self.export_path / f"model.{self.current_task - 1}.pt"
            )
        torch.cuda.empty_cache()
        gc.collect()

    def on_run_end(self) -> None:
        """Used to compute the performance of the ensemble model on the test set."""

    def on_save_checkpoint(self) -> Dict[str, int]:
        return {"current_task": self.current_task}

    def on_load_checkpoint(self, state_dict: Dict) -> None:
        self.current_task = state_dict["current_task"]

    def _reset_fitting(self) -> None:
        # removed as for fast_dev_run it sets _last_val_dl_reload_epoch as 0 and not as -inf, causing 
        # to not reload dataloaders in function _reload_evaluation_dataloaders() 
        #self.trainer.reset_val_dataloader() 

        self.trainer.reset_train_dataloader()
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
