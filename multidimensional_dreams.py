from argparse import ArgumentParser

from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
from executor.cl_loop import BaseCLDataModule, CLLoop
from config.default import datasets
#from loss_function import point_scope

from utils import data_manipulation as datMan
from dataset import dream_sets

from model.SAE import SAE
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
    
    

if __name__ == "__main__":
    args = arg_parser()
    pl.seed_everything(42)

    num_tasks = 5
    num_classes = num_tasks * 10
    epochs_per_task = 15
    dreams_per_target = 48

    fast_dev_run = args.fast_dev_run
    fast_dev_run_batches = False
    if fast_dev_run:
        num_classes = 10
        fast_dev_run_batches = 3
        images_per_dreaming_batch = 8
        dreams_per_target = 16

    dataset_class = getDataset(args.dataset)
    val_tasks_split = train_tasks_split = datMan.classic_tasks_split(num_classes, num_tasks)
    select_dream_tasks_f = datMan.decremental_select_tasks

    def sampler_adapter(dataset, batch_size, shuffle, classes):
        targets = get_target_from_dataset(dataset)

        return PairingBatchSampler(
            dataset=dataset, 
            batch_size=batch_size, 
            main_class_split=0.55,
            shuffle=shuffle,
            classes=classes,
            classes_frequency= [1 / len(classes)] * len(classes)
        )

    sampler_f = sampler_adapter


    #box_border = point_scope.BoxBorder(edge_length=40)
    #loss_object_f = point_scope.PointScopeLoss(border_obj=box_border, dim=num_classes) # default params
    
    loss_object_f = ChiLoss(sigma=0.2)
    dreams_transforms = data_transform()
    model = SAE(
        num_tasks=num_tasks,
        num_classes=num_classes,
        loss_f = loss_object_f
    )    
    #objective_f = datMan.SAE_dream_objective_f
    objective_f = datMan.SAE_multidim_dream_objective_f
    
    #model = VGG11_BN(
    #    num_tasks=num_tasks,
    #    num_classes=num_classes
    #)
    #objective_f = datMan.test

    #from lucent.modelzoo.util import get_model_layers
    #print(get_model_layers(model))
    #exit()
    
    cl_data_module = CLDataModule(
        train_tasks_split=train_tasks_split,
        dataset_class=dataset_class,
        dreams_per_target=dreams_per_target,
        val_tasks_split=val_tasks_split,
        select_dream_tasks_f=select_dream_tasks_f,
        dream_transforms=dreams_transforms,
        fast_dev_run=fast_dev_run,
        datasampler=sampler_f,
        dream_objective_f=objective_f,
        tasks_processing_f=lambda tasks, mean, std : datMan.normall_dist_tasks_processing(tasks, mean=mean, std=std),
        is_multidim = True,
        steps_to_locate_mean=200
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

    cl_data_module.dreams_dataset.dreams[-1].show()