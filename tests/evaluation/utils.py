from datamodule.CLModule import DreamDataModule


class CustomDreamDataModule(DreamDataModule):
    def prepare_data(self):
        raise Exception("Not implemented")

    def train_dataloader(self):
        raise Exception("Not implemented")

    def clear_dreams_dataset(self):
        raise Exception("Not implemented")

    def val_dataloader(self):
        raise Exception("Not implemented")

    def test_dataloader(self):
        raise Exception("Not implemented")

    def setup_tasks(self) -> None:
        raise Exception("Not implemented")

    def setup_task_index(self, task_index: int, loop_index: int) -> None:
        raise Exception("Not implemented")

    def next(self):
        raise Exception("Not implemented")