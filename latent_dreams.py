from argparse import ArgumentParser

from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import RichProgressBar
from executor.cl_loop import BaseCLDataModule, CLLoop
from config.default import datasets, datasets_map
from model.overlay import CLModelWithReconstruction, CLModelWithIslands, CLModel, CLModelIslandsTest
#from loss_function import point_scope

from utils import data_manipulation as datMan
from dataset import dream_sets

from model.SAE import SAE_standalone, SAE_CIFAR, SAE_CIFAR_TEST
from model.vgg import vgg11_bn
from dataset.CLModule import CLDataModule
from dataset.pairing import PairingBatchSampler, PairingBatchSamplerV2
from utils.data_manipulation import get_target_from_dataset
import torch
import numpy as np
from loss_function.chiLoss import ChiLoss

from stats.point_plot import PointPlot, Statistics
from torch.utils.data import DataLoader

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
        "islands_test": CLModelIslandsTest
    }
    return model_types.get(mtype, CLModel)

def get_one_hots(mytype, one_hot_scale=1):
    if(mytype == 'one_cl'):
        return {
            0: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            1: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            2: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            3: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            4: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            5: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            6: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            7: torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) * one_hot_scale,
            8: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            9: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        }
    elif(mytype == 'diagonal'):
        return {
            0: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            1: torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            2: torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            3: torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            4: torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) * one_hot_scale,
            5: torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]) * one_hot_scale,
            6: torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) * one_hot_scale,
            7: torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) * one_hot_scale,
            8: torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]) * one_hot_scale,
            9: torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) * one_hot_scale,
        }
    elif(mytype == 'accidental'):
        return {
            0: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            1: torch.tensor([0, 1, 1, 1, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            2: torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
            3: torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, -1, 0]) * one_hot_scale,
            4: torch.tensor([0, 0, 0, 1, -1, 0, 0, 0, 0, 0]) * one_hot_scale,
            5: torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]) * one_hot_scale,
            6: torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) * one_hot_scale,
            7: torch.tensor([0, 0, 0, 0, 0, 0, -1, -1, -1, 0]) * one_hot_scale,
            8: torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]) * one_hot_scale,
            9: torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 1]) * one_hot_scale,
        }
    
def second_demo():
    # normal dreaming
    args = arg_parser()
    pl.seed_everything(42)

    fast_dev_run = args.fast_dev_run
    fast_dev_run_batches = False

    num_tasks = 1
    num_classes_dataset = 10
    num_classes = 10
    epochs_per_task = 100
    dreams_per_target = 64
    main_split = 0.1
    sigma = 0.2
    rho = 1.
    hidden = 10
    norm_lambd = 0.
    wandb_offline = False
    collect_points = 2500
    nrows = 4
    ncols = 4
    my_batch_size = 32

    train_with_logits = True
    train_normal_robustly = False
    train_dreams_robustly = False
    aux_task_type = "islands"

    only_one_hot = False
    one_hot_means = get_one_hots('diagonal')

    if(fast_dev_run):
        fast_dev_run_batches = 30 # change it to increase epoch count
        #num_tasks = 3
        #num_classes = 6
        #images_per_dreaming_batch = 8
        #epochs_per_task = 2
        #dreams_per_target = 64

    attack_kwargs = {
        "constraint": "2",
        "eps": 0.5,
        "step_size": 1.5,
        "iterations": 10,
        "random_start": 0,
        "custom_loss": None,
        "random_restarts": 0,
        "use_best": True,
        'with_latent': False,
        'fake_relu': False,
        'no_relu':False,
    }

    dream_dataset_class = dream_sets.DreamDatasetWithLogits if train_with_logits else dream_sets.DreamDataset
    dataset_class = getDataset(args.dataset)
    val_tasks_split = train_tasks_split = datMan.classic_tasks_split(num_classes, num_tasks)
    select_dream_tasks_f = datMan.decremental_select_tasks
    dataset_class_robust = getDatasetList(args.dataset)[1]
    model_overlay = getModelType(aux_task_type)
    
    dreams_transforms = data_transform()

    #if fast_dev_run:
    #    val_tasks_split = train_tasks_split = [[0, 1], [2, 3], [4, 5]]

    dataset_robust = dataset_class_robust(data_path="./data", num_classes=num_classes_dataset)

    check(train_tasks_split, num_classes, num_tasks)

    model = model_overlay(
        model=SAE_CIFAR_TEST(num_classes=num_classes, hidd2=hidden),
        #model=vgg11_bn(num_classes=hidden),
        robust_dataset=dataset_robust,
        num_tasks=num_tasks,
        num_classes=num_classes,
        attack_kwargs=attack_kwargs,
        dreams_with_logits=train_with_logits,
        train_normal_robustly=train_normal_robustly,
        #train_dreams_robustly=train_dreams_robustly,
        dream_frequency=10,
        sigma=sigma,
        rho=rho,
        norm_lambd=norm_lambd,
        hidden=hidden,
        one_hot_means=one_hot_means,
        only_one_hot=only_one_hot,
    )

    objective_f = datMan.SAE_dream_objective_f

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
    logger = WandbLogger(project="continual_dreaming", tags=tags, offline=wandb_offline)
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
        #empty_dream_dataset=dream_dataset_class(transform=dreams_transforms),
        progress_bar=progress_bar,
        datasampler=lambda dataset, batch_size, shuffle, classes: 
            #PairingBatchSampler(
            #    dataset=dataset,
            #    batch_size=batch_size,
            #    shuffle=shuffle,
            #    classes=classes,
            #    main_class_split=main_split,
            #    classes_frequency=[1 / len(classes)] * len(classes)
            #),
            PairingBatchSamplerV2(
                dataset=dataset,
                batch_size=batch_size,
                shuffle=shuffle,
                classes=classes,
                main_class_split=main_split,
            ),
        batch_size=my_batch_size
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

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = dataset_class(root="./data", train=True, transform=transform)
    collect_stats(model=model, dataset=dataset, collect_points=collect_points, nrows=nrows, ncols=ncols, 
        logger=logger, attack_kwargs=attack_kwargs)

    # show dream png
    #cl_data_module.dreams_dataset.dreams[-1].show()

def collect_stats(model, dataset, collect_points, attack_kwargs, nrows=1, ncols=1, logger=None):
    stats = Statistics()
    dataloader = DataLoader(dataset, 
        batch_size=32, 
        num_workers=4, 
        pin_memory=False,
    )

    def invoker(model, input):
        xe = model.forward(
            input,
            **attack_kwargs
        )
        return xe
        
    buffer = stats.collect(model=model, dataloader=dataloader, num_of_points=collect_points, to_invoke=invoker,
        logger=logger)
    plotter = PointPlot()
    name = 'plots/multi'

    #plotter.plot(buffer, plot_type='singular', name='plots/singular', show=False, symetric=False, markersize=3, ftype='png')
    #std_mean_dict = Statistics.by_class_operation(Statistics.f_mean_std, buffer, 'saves/mean_std.txt')
    std_mean_distance_dict = Statistics.by_class_operation(Statistics.f_distance, buffer, 'saves/distance.txt')
    std_mean_distance_dict = Statistics.by_class_operation(
        Statistics.f_average_point_dist_from_means, 
        buffer, 
        'saves/average_point_dist_from_means.txt', 
        output=std_mean_distance_dict
    )
    Statistics.mean_distance(std_mean_distance_dict)
    plotter.plot_3d(buffer, std_mean_distance_dict, name='plots/point-plot-3d')
    plotter.plot_std_mean(std_mean_distance_dict, name='plots/std-mean', show=False, ftype='png')
    plotter.plot_distance(std_mean_distance_dict, nrows=nrows, ncols=ncols, name='plots/distance_class', show=False, ftype='png', markersize=3)
    plotter.plot_mean_distance(std_mean_distance_dict, name='plots/mean_distance', show=False, ftype='png', markersize=4)
    plotter.plot_mean_dist_matrix(std_mean_distance_dict, name='plots/mean_dist_matrix', show=False)
    plotter.saveBuffer(buffer, name='saves/latent')

def check(split, num_classes, num_tasks):
    test = set()
    for s in split:
        test = test.union(s)
    if(len(test) != num_classes):
        raise Exception(f"Wrong number of classes: {num_classes} / train or validation split: {len(test)}.")
    if(len(split) != num_tasks):
        raise Exception(f"Wrong number of tasks: {num_tasks} / train or validation split size: {len(split)}.")

if __name__ == "__main__":
    second_demo()