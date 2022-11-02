from argparse import ArgumentParser

from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils.fun_config_set import FunConfigSet, FunConfigSetPredefined
from utils.progress_bar import CustomRichProgressBar
from executor.cl_loop import BaseCLDataModule, CLLoop
from config.default import datasets, datasets_map
from model.overlay import CLModelWithIslands, CLModel, CLModelIslandsTest
#from loss_function import point_scope

from utils import data_manipulation as datMan
from utils.functional import dream_objective
from utils.functional import select_task
from utils.functional import target_processing
from utils.functional import task_split
from dataset import dream_sets

from lucent.optvis import param
from lucent.optvis import transform as tr

from model.ResNet import ResNet18, Resnet20C100
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

from tests.evaluation.compare_latent import CompareLatent
from tests.evaluation.disorder_dream import DisorderDream

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
        "clmodelisland": CLModelWithIslands,
        "clmodel": CLModel,
        "clmodelislandtest": CLModelIslandsTest
    }
    return model_types.get(mtype, CLModel)

def get_one_hots(mytype, one_hot_scale=1, size=10, special_class=1):
    if(mytype == 'one_cl'):
        #d = {
        #    0: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    1: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    2: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    3: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    4: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    5: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    6: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    7: torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) * one_hot_scale,
        #    8: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    9: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #}
        d = {}
        for i in range(size):
            d[i] = torch.zeros((size,), dtype=torch.int32)
        for k, v in d.items():
            v[0] = 1 * one_hot_scale
            if(k == special_class):
                v[size-1] = 1 * one_hot_scale
        return d
    elif(mytype == 'diagonal'):
        #return {
        #    0: torch.tensor([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    1: torch.tensor([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    2: torch.tensor([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    3: torch.tensor([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    4: torch.tensor([0, 0, 0, 0, 1, 0, 0, 0, 0, 0]) * one_hot_scale,
        #    5: torch.tensor([0, 0, 0, 0, 0, 1, 0, 0, 0, 0]) * one_hot_scale,
        #    6: torch.tensor([0, 0, 0, 0, 0, 0, 1, 0, 0, 0]) * one_hot_scale,
        #    7: torch.tensor([0, 0, 0, 0, 0, 0, 0, 1, 0, 0]) * one_hot_scale,
        #    8: torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 1, 0]) * one_hot_scale,
        #    9: torch.tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]) * one_hot_scale,
        #}
        d = {}
        for i in range(size):
            d[i] = torch.zeros((size,), dtype=torch.int32)
            d[i][i] = 1 * one_hot_scale
        return d

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
    
def get_dream_optim():
    def inner(params):
        return torch.optim.Adam(params, lr=5e-3)
    return inner

def param_f_create(ptype, decorrelate=True):

    def param_f_image(image_size, target_images_per_dreaming_batch, **kwargs):
        def param_f():
            # uses 2D Fourier coefficients
            # sd - scale of the random numbers [0, 1)
            return param.image(
                image_size, batch=target_images_per_dreaming_batch, sd=0.4, 
                fft=decorrelate, decorrelate=decorrelate
            )
        return param_f
        
    def param_f_cppn(image_size, **kwargs):
        def param_f():
            return param.cppn(image_size)
        return param_f

    
    if(ptype == 'image'):
        return param_f_image
    elif(ptype == 'cppn'):
        return param_f_cppn
    else:
        raise Exception(f"Unknown type {ptype}")

def CIFAR10_labels():
    return {
        0 : 'airplane',
        1 : 'automobile',
        2 : 'bird',
        3 : 'cat',
        4 : 'deer',
        5 : 'dog',
        6 : 'frog',
        7 : 'horse',
        8 : 'ship',
        9 : 'truck',
    }

def CIFAR100_labels():
    return {
    0: "apple",
    1: "aquarium_fish",
    2: "baby",
    3: "bear",
    4: "beaver",
    5: "bed",
    6: "bee",
    7: "beetle",
    8: "bicycle",
    9: "bottle",
    10: "bowl",
    11: "boy",
    12: "bridge",
    13: "bus",
    14: "butterfly",
    15: "camel",
    16: "can",
    17: "castle",
    18: "caterpillar",
    19: "cattle",
    20: "chair",
    21: "chimpanzee",
    22: "clock",
    23: "cloud",
    24: "cockroach",
    25: "couch",
    26: "crab",
    27: "crocodile",
    28: "cup",
    29: "dinosaur",
    30: "dolphin",
    31: "elephant",
    32: "flatfish",
    33: "forest",
    34: "fox",
    35: "girl",
    36: "hamster",
    37: "house",
    38: "kangaroo",
    39: "keyboard",
    40: "lamp",
    41: "lawn_mower",
    42: "leopard",
    43: "lion",
    44: "lizard",
    45: "lobster",
    46: "man",
    47: "maple_tree",
    48: "motorcycle",
    49: "mountain",
    50: "mouse",
    51: "mushroom",
    52: "oak_tree",
    53: "orange",
    54: "orchid",
    55: "otter",
    56: "palm_tree",
    57: "pear",
    58: "pickup_truck",
    59: "pine_tree",
    60: "plain",
    61: "plate",
    62: "poppy",
    63: "porcupine",
    64: "possum",
    65: "rabbit",
    66: "raccoon",
    67: "ray",
    68: "road",
    69: "rocket",
    70: "rose",
    71: "sea",
    72: "seal",
    73: "shark",
    74: "shrew",
    75: "skunk",
    76: "skyscraper",
    77: "snail",
    78: "snake",
    79: "spider",
    80: "squirrel",
    81: "streetcar",
    82: "sunflower",
    83: "sweet_pepper",
    84: "table",
    85: "tank",
    86: "telephone",
    87: "television",
    88: "tiger",
    89: "tractor",
    90: "train",
    91: "trout",
    92: "tulip",
    93: "turtle",
    94: "wardrobe",
    95: "whale",
    96: "willow_tree",
    97: "wolf",
    98: "woman",
    99: "worm",
}

def second_demo():
    # normal dreaming
    args = arg_parser()
    pl.seed_everything(42)

    fast_dev_run = args.fast_dev_run
    fast_dev_run_batches = 30
    fast_dev_run_epochs = 1
    fast_dev_run_dream_threshold = 32

    num_loops = num_tasks = 1
    num_loops = 1
    scheduler_steps = (1, 6)
    #scheduler_steps = None
    num_classes = 3
    epochs_per_task = 5
    dreams_per_target = 64
    const_target_images_per_dreaming_batch = 8
    main_split = collect_main_split = 0.5
    sigma = 0.01
    rho = 1.
    hidden = 3
    norm_lambd = 0.
    dream_threshold = (512, )
    dream_frequency = 1
    wandb_offline = False if not fast_dev_run else True
    #wandb_offline = True
    enable_dreams = False
    dream_only_once = False # multitask, dream once and do test, exit; sanity check for dreams
    freeze_task_at_end = True
    only_dream_batch = False
    with_reconstruction = False
    run_without_training = False
    collect_numb_of_points = 2500
    cyclic_latent_buffer_size_per_class = 40
    optimizer = lambda param: torch.optim.Adam(param, lr=1e-3)
    #optimizer = lambda param: torch.optim.SGD(param, lr=1e-9, momentum=0.1, weight_decay=0.1)
    #scheduler = None
    scheduler = lambda optim: torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1)
    nrows = 4
    ncols = 4
    my_batch_size = 32
    num_sanity_val_steps = 0
    dream_optim = get_dream_optim()
    train_data_transform = data_transform() #transforms.Compose([transforms.ToTensor()])
    dreams_transforms = data_transform()
    #source_model = SAE_CIFAR(num_classes=num_classes, last_hidd_layer=hidden, with_reconstruction=with_reconstruction)
    #source_model = vgg11_bn(num_classes=hidden)
    #source_model = ResNet18(num_classes=hidden)
    #source_model = Resnet20C100()

    train_with_logits = False
    train_normal_robustly = True
    train_dreams_robustly = True
    aux_task_type = "clmodelislandtest"
    dataset_class_labels = CIFAR100_labels()

    only_one_hot = False
    one_hot_means = get_one_hots(mytype='diagonal', size=hidden)
    clModel_default_loss_f = torch.nn.CrossEntropyLoss()
    param_f = param_f_create(ptype='image')
    #render_transforms = [
    #    tr.pad(4), 
    #    tr.jitter(2), 
    #    tr.random_scale([n/100. for n in range(80, 120)]),
    #    tr.random_rotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2))),
    #    tr.jitter(2),
    #]
    JITTER = 1.1
    ROTATE = 5
    SCALE = 1.1
    render_transforms = [
        tr.pad(2*JITTER),
        tr.jitter(JITTER),
        tr.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
        tr.random_rotate(range(-ROTATE, ROTATE+1))
    ]

    data_passer = {}

    if fast_dev_run:
        tags = ["fast_dev_run"]
    logger = WandbLogger(project="continual_dreaming", tags=tags, offline=wandb_offline)
    progress_bar = CustomRichProgressBar()
    callbacks = [progress_bar]

    #set_manager = FunConfigSet(
    #    select_task_type='select-decremental',
    #    target_processing_type='target-latent-decode',
    #    task_split_type='split-decremental',
    #    dream_obj_type='objective-latent-lossf-creator',
    #    logger=logger,
    #    mtype='sae',
    #    otype='cl-model-island-test',
    #)

    set_manager = FunConfigSetPredefined('decode', logger)

    source_model = set_manager.model(num_classes=num_classes, last_hidd_layer=hidden, with_reconstruction=with_reconstruction)
    target_processing_f = set_manager.target_processing
    select_dream_tasks_f = set_manager.select_task
    objective_f = set_manager.dream_objective
    val_tasks_split = train_tasks_split = set_manager.task_split(num_classes, num_tasks)


    #target_processing_f = target_processing.target_processing_latent_sample_normal_buffer
    #target_processing_f = target_processing.target_processing_latent_buffer_last_point
    #   target_processing_f = target_processing.target_processing_latent_decode
    #   select_dream_tasks_f = select_task.select_task_decremental
    #   objective_f = dream_objective.dream_objective_latent_lossf_creator(logger=logger)
    #objective_f = dream_objective.dream_objective_latent_neuron_direction

    # cross_entropy_loss test
    #target_processing_f = target_processing.target_processing_default
    #select_dream_tasks_f = select_task.select_task_decremental
    #objective_f = dream_objective.dream_objective_channel

    # pretrined RESNET test
    #target_processing_f = target_processing.target_processing_default
    #select_dream_tasks_f = select_task.select_task_decremental
    #objective_f = dream_objective.dream_objective_RESNET20_C100_channel


    if(fast_dev_run):
        pass
        #num_tasks = 3
        #num_classes = 6
        #const_target_images_per_dreaming_batch = 8
        #epochs_per_task = 2
        dreams_per_target = 64

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
    dataset_class_robust = getDatasetList(args.dataset)[1]
    #model_overlay = getModelType(aux_task_type)

    #if fast_dev_run:
    #    val_tasks_split = train_tasks_split = [[0, 1], [2, 3], [4, 5]]

    dataset_robust = dataset_class_robust(data_path="./data")

    check(split=train_tasks_split, 
        num_classes=num_classes, 
        hidden=hidden,
        num_tasks=num_tasks,
        with_reconstruction=with_reconstruction, 
        train_normal_robustly=train_normal_robustly, 
        train_dreams_robustly=train_dreams_robustly,
        aux_task_type=aux_task_type,
    )

    model = set_manager.model_overlay(
        model=source_model,
        robust_dataset=dataset_robust,
        num_tasks=num_tasks,
        num_classes=num_classes,
        attack_kwargs=attack_kwargs,
        dreams_with_logits=train_with_logits,
        train_normal_robustly=train_normal_robustly,
        #train_dreams_robustly=train_dreams_robustly,
        dream_frequency=dream_frequency,
        sigma=sigma,
        rho=rho,
        norm_lambd=norm_lambd,
        hidden=hidden,
        one_hot_means=one_hot_means,
        only_one_hot=only_one_hot,
        cyclic_latent_buffer_size_per_class=cyclic_latent_buffer_size_per_class,
        loss_f=clModel_default_loss_f,
        data_passer=data_passer,
        only_dream_batch=only_dream_batch,
        optimizer_construct_f=optimizer,
        scheduler_construct_f=scheduler,
        scheduler_steps=scheduler_steps,
    )
    print(f'MODEL TYPE: {model.get_obj_str_type()}')

    #from lucent.modelzoo.util import get_model_layers
    #print(get_model_layers(model))
    #exit()
    

    cl_data_module = CLDataModule(
        data_transform=train_data_transform,
        train_tasks_split=train_tasks_split,
        dataset_class=dataset_class,
        dreams_per_target=dreams_per_target,
        val_tasks_split=val_tasks_split,
        select_dream_tasks_f=select_dream_tasks_f,
        param_f=param_f,
        render_transforms=render_transforms,
        fast_dev_run=fast_dev_run,
        fast_dev_run_dream_threshold=fast_dev_run_dream_threshold,
        dream_threshold=dream_threshold,
        dream_objective_f=objective_f,
        empty_dream_dataset=dream_dataset_class(transform=dreams_transforms),
        progress_bar=progress_bar,
        target_processing_f=target_processing_f,
        const_target_images_per_dreaming_batch=const_target_images_per_dreaming_batch,
        optimizer=dream_optim,
        freeze_task_at_end=freeze_task_at_end,
        logger=logger,
        dataset_class_labels=dataset_class_labels,
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
        #fast_dev_run=fast_dev_run_batches, # error when multiple tasks - in new task 0 batches are done.
        limit_train_batches=fast_dev_run_batches if fast_dev_run else None,
        gpus="0,",
        log_every_n_steps=1 if fast_dev_run else 50,
        num_sanity_val_steps=num_sanity_val_steps,
    )
    progress_bar._init_progress(trainer)
    

    print(f"Fast dev run is {fast_dev_run}")

    internal_fit_loop = trainer.fit_loop
    trainer.fit_loop = CLLoop(
        [epochs_per_task] * num_tasks, 
        enable_dreams=enable_dreams,
        fast_dev_run_epochs=fast_dev_run_epochs,
        fast_dev_run=fast_dev_run,
        data_passer=data_passer,
        num_loops=num_loops,
        run_without_training=run_without_training,
        dream_only_once=dream_only_once,
    )
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule=cl_data_module)

    trainer.test(model, datamodule=cl_data_module)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = dataset_class(root="./data", train=False, transform=transform)
    dataset = CLDataModule._split_dataset(dataset, [np.concatenate(train_tasks_split, axis=0)])[0]
    targets = None
    if isinstance(dataset, torch.utils.data.Subset):
        targets = np.take(dataset.dataset.targets, dataset.indices).tolist()
    else:
        targets = dataset.targets
    collector_batch_sampler = PairingBatchSamplerV2(
                dataset=dataset,
                batch_size=my_batch_size,
                shuffle=True,
                classes=np.unique(targets),
                main_class_split=collect_main_split,
            )
    #collect_stats(model=model, dataset=dataset,
    #    collect_numb_of_points=collect_numb_of_points, 
    #    collector_batch_sampler=collector_batch_sampler,
    #    nrows=nrows, ncols=ncols, 
    #    logger=logger, attack_kwargs=attack_kwargs)

    #compare_latent = CompareLatent()
    #compare_latent(
    #    model=model,
    #    loss_f=model.loss_f, 
    #    used_class=1, 
    #    logger=logger,
    #    dream_transform=dreams_transforms,
    #)

    disorder_dream = DisorderDream()
    dataset = dataset_class(root="./data", train=False, transform=transform)
    disorder_dream(
        model=model,
        loss_f=model.loss_f, 
        used_class=1, 
        logger=logger,
        dataset=dataset,
        dream_transform=dreams_transforms,
        batch_size=my_batch_size,
    )

    # show dream png
    #cl_data_module.dreams_dataset.dreams[-1].show()

def collect_stats(model, dataset, collect_numb_of_points, collector_batch_sampler, attack_kwargs, nrows=1, ncols=1, logger=None):
    stats = Statistics()
    dataloader = DataLoader(dataset, 
        num_workers=4, 
        pin_memory=False,
        batch_sampler=collector_batch_sampler
    )

    def invoker(model, input):
        xe = model.forward(
            input,
            **attack_kwargs
        )
        if(isinstance(xe, tuple)):
            xe = xe[0]
        return xe
        
    buffer = stats.collect(model=model, dataloader=dataloader, num_of_points=collect_numb_of_points, to_invoke=invoker,
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

def check(split, num_classes, hidden, num_tasks, with_reconstruction, train_normal_robustly, train_dreams_robustly, aux_task_type):
    test = set()
    for s in split:
        test = test.union(s)
    if(len(test) != num_classes):
        raise Exception(f"Wrong number of classes: {num_classes} / train or validation split: {len(test)}.")
    if(len(split) != num_tasks):
        raise Exception(f"Wrong number of tasks: {num_tasks} / train or validation split size: {len(split)}.")
    if(with_reconstruction and (train_normal_robustly or train_dreams_robustly)):
        raise Exception(f"Framework robusness does not support model that returns multiple variables. Set correct flags. Current:\
\nwith_reconstruction: {with_reconstruction}\ntrain_normal_robustly: {train_normal_robustly}\ntrain_dreams_robustly: {train_dreams_robustly}")
    if(num_classes > hidden and aux_task_type == 'default'):
        raise Exception(f"Hidden dimension {hidden} is smaller than number of classes {num_classes} in binary classification.")


if __name__ == "__main__":
    second_demo()