from argparse import ArgumentParser

from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
from executor.cl_loop import BaseCLDataModule, CLLoop
from config.default import datasets, datasets_map
from model.overlay import CLModelWithReconstruction, CLModelWithIslands, CLModel
#from loss_function import point_scope

from utils import data_manipulation as datMan
from dataset import dream_sets

from model.SAE import SAE_standalone, SAE_CIFAR
from model.vgg import VGG11_BN
from dataset.CLModule import CLDataModule
from dataset.pairing import PairingBatchSampler
from utils.data_manipulation import get_target_from_dataset
import torch
import numpy as np
from loss_function.chiLoss import ChiLoss

def arg_parser():
    parser = ArgumentParser()
    parser.add_argument("-f", "--fast-dev-run", action="store_true")
    parser.add_argument("-d", "--dataset", type=str)
    return parser.parse_args()

def data_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]
    )

def getDataset(name:str):
    if name is not None:
        cap = name.capitalize()
        if(cap in datasets):
            return datasets[cap]
    raise Exception(f"Unknown dataset {name}.")
    
def getDatasetList(name:str):
    if name is not None:
        cap = name.capitalize()
        if(cap in datasets_map):
            return datasets_map[cap]
    raise Exception(f"Unknown dataset {name}.")

def getModelType(mtype: str):
    model_types = {
        "auxiliary_reconstruction": CLModelWithReconstruction,
        "islands": CLModelWithIslands,
        "default": CLModel,
    }
    return model_types.get(mtype, CLModel)
    
def second_demo():
    # normal dreaming
    args = arg_parser()
    pl.seed_everything(42)

    num_tasks = 5
    num_classes = 10
    epochs_per_task = 15
    dreams_per_target = 48

    train_with_logits = True
    train_normal_robustly = False
    train_dreams_robustly = False
    aux_task_type = "islands"

    attack_kwargs = attack_kwargs = {
        "constraint": "2",
        "eps": 0.5,
        "step_size": 1.5,
        "iterations": 10,
        "random_start": 0,
        "custom_loss": None,
        "random_restarts": 0,
        "use_best": True,
        'with_latent': True,
        'fake_relu': True,
    }

    dream_dataset_class = dream_sets.DreamDatasetWithLogits if train_with_logits else dream_sets.DreamDataset
    dataset_class = getDataset(args.dataset)
    val_tasks_split = train_tasks_split = datMan.classic_tasks_split(num_classes, num_tasks)
    select_dream_tasks_f = datMan.decremental_select_tasks
    dataset_class_robust = getDatasetList(args.dataset)[1]
    model_overlay = getModelType(aux_task_type)
    
    dreams_transforms = data_transform()

    fast_dev_run = args.fast_dev_run
    fast_dev_run_batches = False
    if fast_dev_run:
        num_classes = 4
        fast_dev_run_batches = 10
        images_per_dreaming_batch = 8
        dreams_per_target = 16
        num_tasks = 2
        val_tasks_split = train_tasks_split = [[0, 1], [2, 3]]

    dataset_robust = dataset_class_robust(data_path="./data", num_classes=num_classes)

    model = model_overlay(
        model=SAE_CIFAR(num_classes=num_classes),
        robust_dataset=dataset_robust,
        num_tasks=num_tasks,
        num_classes=num_classes,
        attack_kwargs=attack_kwargs,
        dreams_with_logits=train_with_logits,
        train_normal_robustly=train_normal_robustly,
        train_dreams_robustly=train_dreams_robustly,
    )

    objective_f = datMan.SAE_dream_objective_f


    #model = SAE_standalone(
    #    num_tasks=num_tasks,
    #    num_classes=num_classes,
    #    loss_f = None
    #)    
    #objective_f = datMan.SAE_dream_objective_f
    
    #model = VGG11_BN(
    #    num_tasks=num_tasks,
    #    num_classes=num_classes
    #)
    #objective_f = datMan.test

    #from lucent.modelzoo.util import get_model_layers
    #print(get_model_layers(model))
    #exit()

    tags = []
    if train_with_logits:
        tags.append("logits")
    if train_normal_robustly or train_dreams_robustly:
        tags.append("robust")
    if aux_task_type == "auxiliary_reconstruction":
        tags.append("auxiliary")
    if aux_task_type == "island":
        tags.append("island")
    if fast_dev_run:
        tags = ["fast_dev_run"]
    logger = WandbLogger(project="continual_dreaming", tags=tags)
    progress_bar = RichProgressBar()
    callbacks = [progress_bar]
    

    cl_data_module = CLDataModule(
        train_tasks_split=train_tasks_split,
        dataset_class=dataset_class,
        dreams_per_target=dreams_per_target,
        val_tasks_split=val_tasks_split,
        select_dream_tasks_f=select_dream_tasks_f,
        fast_dev_run=fast_dev_run,
        dream_objective_f=objective_f,
        empty_dream_dataset=dream_dataset_class(transform=dreams_transforms),
        progress_bar=progress_bar,
        datasampler=lambda dataset, batch_size, shuffle, classes: 
            PairingBatchSampler(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                classes=classes,
                main_class_split=0.55,
                classes_frequency=[1 / len(classes)] * len(classes)
            )
    )

    trainer = pl.Trainer(
        max_epochs=-1,  # This value doesn't matter
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=fast_dev_run_batches,
        gpus="0,",
    )

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = CLLoop([epochs_per_task] * num_tasks, progress_bar)
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule=cl_data_module)
    if not fast_dev_run:
        trainer.test(datamodule=cl_data_module)

    # show dream png
    cl_data_module.dreams_dataset.dreams[-1].show()

if __name__ == "__main__":
    second_demo()