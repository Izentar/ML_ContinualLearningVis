import numpy as np
import pytorch_lightning as pl
import torch
import torchmetrics
import wandb
from pytorch_lightning import LightningDataModule, LightningModule
from pytorch_lightning.loggers import WandbLogger
from torch import nn
from torch.nn.functional import cross_entropy
from torch.utils.data import ConcatDataset, DataLoader, TensorDataset, Subset
from torchvision import transforms
from torchvision.transforms.functional import to_pil_image
from lucent.optvis import render, param, objectives
from math import ceil


from robustness import model_utils

from torchvision.datasets import CIFAR100
from torchvision.datasets import CIFAR10
from robustness.datasets import CIFAR100 as CIFAR100_robust
from robustness.datasets import CIFAR10 as CIFAR10_robust


class DreamDataset:
    def __init__(self, transform=None) -> None:
        self.dreams = []
        self.targets = []
        self.transform = transform

    def __len__(self):
        return len(self.dreams)

    def __getitem__(self, idx):
        dream = self.dreams[idx]
        if self.transform:
            dream = self.transform(dream)

        return dream, self.targets[idx]

    def extend(self, new_dreams, new_targets):
        assert len(new_dreams) == len(new_targets)
        for dream in new_dreams:
            if self.transform:
                dream = to_pil_image(dream)
            self.dreams.append(dream)
        self.targets.extend(new_targets)


class CLDataModule(LightningDataModule):
    def __init__(
        self,
        model,
        train_tasks_split,
        epochs_per_dataset,
        dreams_per_target,
        dataset_class,
        images_per_dreaming_batch=8,
        val_tasks_split=None,
        dream_transforms=None,
        max_logged_dreams=8,
    ):
        """
        Args:
            task_split: list containing list of class indices per task
        """
        super().__init__()
        self.train_tasks_split = train_tasks_split
        self.val_tasks_split = (
            val_tasks_split if val_tasks_split is not None else train_tasks_split
        )
        self.curr_index = 0
        self.epochs_per_dataset = epochs_per_dataset
        self.model = model
        self.dreams_per_target = dreams_per_target
        self.images_per_dreaming_batch = images_per_dreaming_batch
        # TODO
        self.image_size = 32
        self.max_logged_dreams = max_logged_dreams
        self.dreams_dataset = DreamDataset(transform=dream_transforms)
        self.dataset_class = dataset_class
        #  self.dreams = torch.tensor([])
        #  self.dreams_targets = torch.tensor([])

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = self.dataset_class(
            root="data", train=True, transform=transform, download=True
        )
        test_dataset = self.dataset_class(root="data", train=False, transform=transform)

        self.train_datasets = self._split_dataset(train_dataset, self.train_tasks_split)
        self.test_datasets = self._split_dataset(test_dataset, self.val_tasks_split)
        self.train_dataset = train_dataset

    @staticmethod
    def _split_dataset(dataset, tasks_split):
        split_dataset = []
        for current_classes in tasks_split:
            task_indices = np.isin(np.array(dataset.targets), current_classes)
            split_dataset.append(Subset(dataset, np.where(task_indices)[0]))
        return split_dataset

    def train_dataloader(self):
        if self.curr_index > 0:
            self.generate_new_dreams()
            #  dreams_dataset = TensorDataset(self.dreams, self.dreams_targets)
            dream_loader = DataLoader(
                self.dreams_dataset, batch_size=32, num_workers=4, shuffle=True
            )
        else:
            dream_loader = None
        normal_loader = DataLoader(
            self.train_datasets[self.curr_index],
            batch_size=32,
            num_workers=4,
            shuffle=True,
        )
        self.curr_index += 1
        if dream_loader is not None:
            return {"normal": normal_loader, "dream": dream_loader}
        else:
            return {"normal": normal_loader}

    def val_dataloader(self):
        return [
            DataLoader(dataset, batch_size=32, num_workers=4)
            for dataset in self.test_datasets
        ]

    def test_dataloader(self):
        return DataLoader(
            ConcatDataset(self.train_datasets), batch_size=32, num_workers=4
        )

    def generate_new_dreams(self):
        current_split = set(self.train_tasks_split[self.curr_index])
        previous_split = set(self.train_tasks_split[self.curr_index - 1])
        diff_targets = current_split.symmetric_difference(previous_split)

        model_mode = self.model.training
        if model_mode:
            self.model = self.model.eval()

        new_dreams = []
        new_targets = []
        for target in diff_targets:
            target_dreams = self.generate_dreams_for_target(target)
            new_targets.extend([target] * target_dreams.shape[0])
            new_dreams.append(target_dreams)
        new_dreams = torch.cat(new_dreams)
        new_targets = torch.tensor(new_targets)
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
        self.dreams_dataset.extend(new_dreams, new_targets)
        #  self.dreams = torch.cat([self.dreams, new_dreams])
        #  self.dreams_targets = torch.cat([self.dreams_targets, new_targets]).long()

        if model_mode:
            self.model = self.model.train()

    def generate_dreams_for_target(self, target):
        dreams = []
        iterations = ceil(self.dreams_per_target / self.images_per_dreaming_batch)
        for _ in range(iterations):

            def batch_param_f():
                return param.image(
                    self.image_size, batch=self.images_per_dreaming_batch,
                    sd=0.4
                )

            obj = objectives.channel(
                "resnet_model_linear", target
            ) - objectives.diversity("resnet_model_layer4")
            dreams.append(
                torch.permute(
                    torch.from_numpy(
                        render.render_vis(
                            self.model,
                            obj,
                            batch_param_f,
                            fixed_image_size=self.image_size,
                            progress=False,
                            show_image=False,
                        )[0]
                    ),
                    (0, 3, 1, 2),
                )
            )
        return torch.cat(dreams)


def classic_tasks_split(num_classes, num_tasks):
    one_split = num_classes // num_tasks
    return [list(range(i * one_split, (i + 1) * one_split)) for i in range(num_tasks)]


def decremental_tasks_split(num_classes, num_tasks):
    return [list(range(num_classes))[i * 2 :] for i in range(num_tasks)]


class CLBaseModel(LightningModule):
    def __init__(self, num_tasks, attack_kwargs):
        super().__init__()

        self.train_acc = torchmetrics.Accuracy()
        self.train_acc_dream = torchmetrics.Accuracy()
        self.valid_accs = nn.ModuleList(
            [torchmetrics.Accuracy() for _ in range(num_tasks)]
        )
        self.test_acc = torchmetrics.Accuracy()

        self.attack_kwargs = attack_kwargs

    def training_step(self, batch, batch_idx):
        batch_normal = batch["normal"]
        loss_normal = self.training_step_normal(batch_normal)
        if "dream" not in batch:
            return loss_normal
        loss_dream = self.training_step_dream(batch["dream"])
        return loss_normal + loss_dream

    def training_step_normal(self, batch):
        x, y = batch
        y_hat, _final_input = self(x, target=y, make_adv=True, **self.attack_kwargs)
        loss = cross_entropy(y_hat, y)
        self.log("train_loss", loss)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def training_step_dream(self, batch):
        x, y = batch
        y_hat = self(x)
        loss = cross_entropy(y_hat, y)
        self.log("train_loss_dream", loss)
        self.train_acc_dream(y_hat, y)
        self.log("train_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat = self(x)
        val_loss = cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(y_hat, y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        test_loss = cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class ResNet18(CLBaseModel):
    def __init__(self, ds, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.resnet, _ = model_utils.make_and_restore_model(arch="resnet18", dataset=ds)

    def forward(self, *args, make_adv=False, **kwargs):
        out = self.resnet(*args, make_adv=make_adv, **kwargs)
        if make_adv:
            return out
        else:
            return out[0]


if __name__ == "__main__":
    num_tasks = 5
    num_classes = 50
    epochs_per_task = 15
    dreams_per_target = 48
    cifar_100 = True  # CIFAR10 if false
    if not cifar_100:
        assert num_classes <= 10
    attack_kwargs = attack_kwargs = {
        "constraint": "2",
        "eps": 0.5,
        "step_size": 1.5,
        "iterations": 10,
        "random_start": 0,
        "custom_loss": None,
        "random_restarts": 0,
        "use_best": True,
    }
    pl.seed_everything(42)
    dreams_transforms = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )
    ds_class = CIFAR100 if cifar_100 else CIFAR10
    ds_class_robust = CIFAR100_robust if cifar_100 else CIFAR10_robust
    ds_robust = ds_class_robust(data_path="./data", num_classes=num_classes)
    train_tasks_split = classic_tasks_split(num_classes, num_tasks)
    val_tasks_split = train_tasks_split
    model = ResNet18(ds=ds_robust, num_tasks=num_tasks, attack_kwargs=attack_kwargs)
    cl_data_module = CLDataModule(
        model,
        train_tasks_split,
        epochs_per_task,
        dataset_class=ds_class,
        dreams_per_target=dreams_per_target,
        val_tasks_split=val_tasks_split,
        dream_transforms=dreams_transforms,
    )
    logger = WandbLogger(project="continual_dreaming", tags=["dummy", "profile"])
    trainer = pl.Trainer(
        max_epochs=num_tasks * epochs_per_task,
        reload_dataloaders_every_n_epochs=epochs_per_task,
        logger=logger,
        gpus="0,",
    )
    trainer.fit(model, datamodule=cl_data_module)
    trainer.test(datamodule=cl_data_module)
