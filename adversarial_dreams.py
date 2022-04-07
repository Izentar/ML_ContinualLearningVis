from argparse import ArgumentParser
from math import ceil

import numpy as np
import pytorch_lightning as pl
import torch
from torch.autograd.variable import Variable
import torchmetrics
import wandb
from lucent.optvis import objectives, param, render
from pytorch_lightning import LightningModule
from pytorch_lightning.callbacks import RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from torch import nn, sigmoid
from torch.nn.functional import cross_entropy, mse_loss, relu
from torch.utils.data import ConcatDataset, DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torchvision.transforms.functional import to_pil_image

from cl_loop import BaseCLDataModule, CLLoop
#  from robustness import model_utils
from robustness.datasets import CIFAR10 as CIFAR10_robust
from robustness.datasets import CIFAR100 as CIFAR100_robust


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


class CLDataModule(BaseCLDataModule):
    def __init__(
        self,
        train_tasks_split,
        dreams_per_target,
        dataset_class,
        images_per_dreaming_batch=8,
        val_tasks_split=None,
        dream_transforms=None,
        max_logged_dreams=8,
        fast_dev_run=False,
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
        self.dreams_per_target = dreams_per_target
        self.images_per_dreaming_batch = images_per_dreaming_batch
        if fast_dev_run:
            self.images_per_dreaming_batch = 8
            self.dreams_per_target = 16
        self.fast_dev_run = fast_dev_run
        # TODO
        self.image_size = 32
        self.max_logged_dreams = max_logged_dreams
        self.dreams_dataset = DreamDataset(transform=dream_transforms)
        self.dataset_class = dataset_class

    def prepare_data(self):
        transform = transforms.Compose([transforms.ToTensor()])
        train_dataset = self.dataset_class(
            root="data", train=True, transform=transform, download=True
        )
        test_dataset = self.dataset_class(root="data", train=False, transform=transform)

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

    def setup_task_index(self, task_index: int) -> None:
        self.train_task = self.train_datasets[task_index]
        self.dream_task = self.dreams_dataset if len(self.dreams_dataset) > 0 else None

    def train_dataloader(self):
        dream_loader = None
        if self.dream_task:
            dream_loader = DataLoader(
                self.dream_task, batch_size=32, num_workers=4, shuffle=True
            )
        normal_loader = DataLoader(
            self.train_task,
            batch_size=32,
            num_workers=4,
            shuffle=True,
        )
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

    def generate_synthetic_data(self, model, task_index):
        """Generate new dreams."""
        current_split = set(self.train_tasks_split[task_index])
        previous_split = set(self.train_tasks_split[task_index - 1])
        diff_targets = previous_split - current_split

        model_mode = model.training
        if model_mode:
            model = model.eval()

        new_dreams = []
        new_targets = []
        iterations = ceil(self.dreams_per_target / self.images_per_dreaming_batch)
        with Progress(
            "[progress.description]{task.description}",
            BarColumn(complete_style="magenta"),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            dreaming_progress = progress.add_task(
                "[bright_blue]Dreaming...", total=(len(diff_targets) * iterations)
            )
            for target in sorted(diff_targets):
                target_progress = progress.add_task(
                    f"[bright_red]Class: {target}", total=iterations
                )

                def update_progress():
                    progress.update(target_progress, advance=1)
                    progress.update(dreaming_progress, advance=1)

                target_dreams = self._generate_dreams_for_target(
                    model, target, iterations, update_progress
                )
                new_targets.extend([target] * target_dreams.shape[0])
                new_dreams.append(target_dreams)
                progress.remove_task(target_progress)
        new_dreams = torch.cat(new_dreams)
        new_targets = torch.tensor(new_targets)
        if not self.fast_dev_run:
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

        if model_mode:
            model = model.train()

    def _generate_dreams_for_target(self, model, target, iterations, update_progress):
        dreams = []
        for _ in range(iterations):

            def batch_param_f():
                return param.image(
                    self.image_size, batch=self.images_per_dreaming_batch, sd=0.4
                )

            obj = objectives.channel("sae_fc", target) - objectives.diversity(
                "sae_conv2"
            )
            dreams.append(
                torch.permute(
                    torch.from_numpy(
                        render.render_vis(
                            model,
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
            update_progress()
        return torch.cat(dreams)


def classic_tasks_split(num_classes, num_tasks):
    one_split = num_classes // num_tasks
    return [list(range(i * one_split, (i + 1) * one_split)) for i in range(num_tasks)]


def decremental_tasks_split(num_classes, num_tasks):
    return [list(range(num_classes))[i * 2 :] for i in range(num_tasks)]


class CLBaseModel(LightningModule):
    def __init__(self, num_tasks, num_classes, attack_kwargs):
        super().__init__()

        self.train_acc = torchmetrics.Accuracy()
        self.train_acc_dream = torchmetrics.Accuracy()
        self.valid_accs = nn.ModuleList(
            [torchmetrics.Accuracy() for _ in range(num_tasks)]
        )
        self.test_acc = torchmetrics.Accuracy()
        self.num_classes = num_classes

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
        y_hat, y_reconstruction = self(x)
        loss_classification = cross_entropy(y_hat, y)
        loss_reconstruction = mse_loss(y_reconstruction, Variable(x))
        alpha = 0.05
        loss = alpha * loss_classification + (1 - alpha) * loss_reconstruction
        self.log("train_loss/total", loss)
        self.log("train_loss/classification", loss_classification)
        self.log("train_loss/reconstrucion", loss_reconstruction)
        self.train_acc(y_hat, y)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return loss

    def training_step_dream(self, batch):
        x, y = batch
        y_hat, y_reconstruction = self(x)
        loss_classification = cross_entropy(y_hat, y)
        loss_reconstruction = mse_loss(y_reconstruction, Variable(x))
        alpha = 0.05
        loss = alpha * loss_classification + (1 - alpha) * loss_reconstruction
        self.log("train_loss_dream/total", loss)
        self.log("train_loss_dream/classification", loss_classification)
        self.log("train_loss_dream/reconstrucion", loss_reconstruction)
        self.train_acc_dream(y_hat, y)
        self.log("train_acc_dream", self.train_acc_dream, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        y_hat, _ = self(x)
        val_loss = cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)
        valid_acc = self.valid_accs[dataloader_idx]
        valid_acc(y_hat, y)
        self.log("valid_acc", valid_acc)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat, _ = self(x)
        test_loss = cross_entropy(y_hat, y)
        self.log("test_loss", test_loss)
        self.test_acc(y_hat, y)
        self.log("test_acc", self.test_acc)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)


class SAE_CIFAR(nn.Module):
    def __init__(self, num_classes, hidd1=256, hidd2=32, multi_head=False):
        super(SAE_CIFAR, self).__init__()
        self.multi_head = multi_head
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
        )
        self.fc1_2 = nn.Linear(in_features=50176, out_features=hidd1)
        self.fc2_3 = nn.Linear(in_features=hidd1, out_features=hidd2)

        self.fc3_2 = nn.Linear(in_features=hidd2, out_features=hidd1)
        self.fc2_1 = nn.Linear(in_features=hidd1, out_features=50176)
        self.conv3 = nn.ConvTranspose2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
        )
        self.conv4 = nn.ConvTranspose2d(
            in_channels=64, out_channels=3, kernel_size=(3, 3), stride=(1, 1)
        )

        self.fc = nn.Linear(in_features=hidd2, out_features=num_classes)

    def forward(self, x):
        xe = relu(self.conv1(x))
        xe = relu(self.conv2(xe))
        shp = [xe.shape[0], xe.shape[1], xe.shape[2], xe.shape[3]]
        xe = xe.view(-1, shp[1] * shp[2] * shp[3])
        xe = relu(self.fc1_2(xe))
        xe = relu(self.fc2_3(xe))

        xd = relu(self.fc3_2(xe))
        xd = relu(self.fc2_1(xd))
        xd = torch.reshape(xd, (shp[0], shp[1], shp[2], shp[3]))
        xd = relu(self.conv3(xd))
        # xd = F.upsample(xd,30)
        x_hat = sigmoid(self.conv4(xd))

        if self.multi_head:
            return xe, x_hat
        else:
            y_hat = self.fc(xe)
            return y_hat, x_hat


class SAE(CLBaseModel):
    def __init__(self, num_classes, *args, **kwargs):
        super().__init__(*args, num_classes=num_classes, **kwargs)

        self.sae = SAE_CIFAR(num_classes)

    def forward(self, *args):
        return self.sae(*args)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-f", "--fast-dev-run", action="store_true")
    args = parser.parse_args()
    fast_dev_run = args.fast_dev_run
    cifar_100 = True  # CIFAR10 if false
    num_tasks = 5
    num_classes = 50
    epochs_per_task = 15
    dreams_per_target = 48
    fast_dev_run_batches = False
    if fast_dev_run:
        num_classes = 10
        fast_dev_run_batches = 3
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
    model = SAE(
        num_tasks=num_tasks,
        num_classes=num_classes,
        attack_kwargs=attack_kwargs,
    )
    cl_data_module = CLDataModule(
        train_tasks_split,
        dataset_class=ds_class,
        dreams_per_target=dreams_per_target,
        val_tasks_split=val_tasks_split,
        dream_transforms=dreams_transforms,
        fast_dev_run=fast_dev_run,
    )
    tags = [] if not fast_dev_run else ["debug"]
    logger = WandbLogger(project="continual_dreaming", tags=tags)
    callbacks = [RichProgressBar()]
    trainer = pl.Trainer(
        max_epochs=-1,  # This value doesn't matter
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=fast_dev_run_batches,
        gpus="0,",
    )
    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = CLLoop([epochs_per_task] * num_tasks)
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule=cl_data_module)
    if not fast_dev_run:
        trainer.test(datamodule=cl_data_module)
