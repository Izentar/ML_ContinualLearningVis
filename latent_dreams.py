from torchvision import transforms
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from utils.fun_config_set import FunConfigSet, FunConfigSetPredefined
from utils.progress_bar import CustomRichProgressBar
from executor.cl_loop import CLLoop
from config.default import datasets, datasets_map
#from loss_function import point_scope

from dataset import dream_sets

from lucent.optvis import param
from dream import transform as dream_tr

from dataset.CLModule import CLDataModule
from dataset.pairing import PairingBatchSampler, PairingBatchSamplerV2
import torch
import numpy as np

from stats.point_plot import PointPlot, Statistics
from torch.utils.data import DataLoader

from tests.evaluation.compare_latent import CompareLatent
from tests.evaluation.disorder_dream import DisorderDream
from my_parser import arg_parser, log_to_wandb, wandb_run_name, export_config, can_export_config

from model.statistics.base import pca
from collections.abc import Sequence
from utils.utils import parse_image_size
import wandb
from dream.image import Image
from utils import utils
from model.activation_layer import GaussA
from pathlib import Path
from model.overlay import CLModel

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
    
def param_f_create(ptype):

    def param_f_image(dtype, image_size, dreaming_batch_size, decorrelate, **kwargs):
        channels, w, h = parse_image_size(image_size)
        # uses 2D Fourier coefficients
        # sd - scale of the random numbers [0, 1)
        return Image(dtype=dtype, w=w, h=h, channels=channels, batch=dreaming_batch_size,
            decorrelate=decorrelate)
        
    def param_f_cppn(image_size, **kwargs):
        print(f'Selected dream image type: cppn')
        def param_f():
            return param.cppn(image_size)
        return param_f

    
    if(ptype == 'fft' or ptype == 'pixel' or ptype == 'custom'):
        return param_f_image
    elif(ptype == 'cppn'):
        return param_f_cppn
    else:
        raise Exception(f"Unknown type {ptype}")

def select_datasampler(dtype, main_split):
    def inner(dataset, batch_size, shuffle, classes): 
        return PairingBatchSamplerV2(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            classes=classes,
            main_class_split=main_split,
        )
    def inner2(dataset, batch_size, shuffle, classes):
        return PairingBatchSampler(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            classes=classes,
            main_class_split=main_split,
            classes_frequency=[1 / len(classes)] * len(classes)
        )

    if(dtype == 'none'):
        print('INFO: Selected None datasampler')
        return None
    elif(dtype == 'v2'):
        print('INFO: Selected PairingBatchSamplerV2 datasampler')
        return inner
    else:
        raise Exception(f'Unknown type: {dtype}')

def model_summary(source_model):
    from torchsummary import summary
    source_model = source_model.cuda()
    summary(source_model, (3, 32, 32))
    exit()

def logic(args, log_args_to_wandb=True):
    # normal dreaming
    if(args.config.seed is not None):
        pl.seed_everything(args.config.seed)

    main_split = collect_main_split = 0.5
    wandb_offline = False if not args.fast_dev_run.enable else True
    wandb_mode = "online" if not args.fast_dev_run.enable else "disabled"
    #wandb_offline = True
    
    num_sanity_val_steps = 0
    train_data_transform = data_transform() #transforms.Compose([transforms.ToTensor()])
    dreams_transforms = data_transform()

    datasampler = select_datasampler(dtype=args.config.datasampler_type, main_split=main_split)

    one_hot_means = get_one_hots(mytype='diagonal', size=args.model.num_classes)
    clModel_default_loss_f = torch.nn.CrossEntropyLoss()
    dream_image_f = param_f_create(ptype=args.datamodule.vis.image_type)
    render_transforms = None
    #render_transforms = [
    #    dream_tr.pad(4), 
    #    dream_tr.jitter(2), 
    #    dream_tr.random_scale([n/100. for n in range(80, 120)]),
    #    dream_tr.random_rotate(list(range(-10,10)) + list(range(-5,5)) + 10*list(range(-2,2))),
    #    dream_tr.jitter(2),
    #]
    JITTER = 2
    ROTATE = 5
    SCALE = 1.1
    #render_transforms = [
    #    dream_tr.pad(JITTER), # int
    #    dream_tr.jitter(JITTER), # > 1
    #    dream_tr.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
    #    dream_tr.random_rotate(range(-ROTATE, ROTATE+1)),
    #    dream_tr.jitter(int(JITTER))
    #]

    render_transforms = [
        dream_tr.pad(2*JITTER),
        dream_tr.jitter(JITTER),
        dream_tr.random_scale([SCALE ** (n/10.) for n in range(-10, 11)]),
        dream_tr.random_rotate(range(-ROTATE, ROTATE+1))
    ]

    data_passer = {}

    tags = []
    wandb_run_id = wandb.util.generate_id()
    if args.fast_dev_run.enable:
        tags = ["fast_dev_run"]
    logger = WandbLogger(project="continual_dreaming", tags=tags, offline=wandb_offline, mode=wandb_mode, name=wandb_run_name(args),
        log_model=False, save_dir=args.wandb.run.folder, config=args, save_code=False, id=wandb_run_id)
    if(log_args_to_wandb):
        log_to_wandb(args)
    progress_bar = CustomRichProgressBar()
    callbacks = [progress_bar]

    #print(wandb_run_id, wandb.run.name, wandb.run.id)
    #wandb.save('tmp.txt')
    #exit()

    #set_manager = FunConfigSet(
    #    select_task_type='select-decremental',
    #    target_processing_type='target-latent-decode',
    #    task_split_type='split-decremental',
    #    dream_obj_type='objective-latent-lossf-creator',
    #    logger=logger,
    #    mtype='sae',
    #    otype='cl-model-island-test',
    #)

    #### setting in dream_obj_type a diversity type will extend time of dreaming significantly
    set_manager = FunConfigSetPredefined(
        name_type=args.config.framework_type, 
        mtype=args.model.type,
        select_task_type=args.config.select_task_type,
        target_processing_type=args.config.target_processing_type,
        task_split_type=args.config.task_split_type,
        otype=args.config.overlay_type,
        #mtype='SAEGAUSS', 
        #mtype='RESNET18', 
        dream_obj_type=args.config.dream_obj_type,
        #dream_obj_type=["objective-channel", "OBJECTIVE-DLA-DIVERSITY-1", "OBJECTIVE-DLA-DIVERSITY-2", "OBJECTIVE-DLA-DIVERSITY-3"],
        logger=logger
    )
    set_manager.init_dream_objectives(logger=logger, label='dream')
    print(f"Selected configuration:\n{str(set_manager)}")

    source_model = set_manager.model(num_classes=args.model.num_classes)
    target_processing_f = set_manager.target_processing
    select_dream_tasks_f = set_manager.select_task
    objective_f = set_manager.dream_objective
    val_tasks_split = train_tasks_split = set_manager.task_split(args.model.num_classes, args.config.num_tasks)

    #model_summary(source_model)

    if(args.model.robust.enable):
        print('WARNING:\tTRAIN ROBUSTLY IS ENABLED, SLOWER TRAINING.')

    if(args.fast_dev_run.enable):
        args.datamodule.vis.per_target = 64

    dataset_class, dataset_class_labels = getDataset(args.config.dataset)

    #if args.fast_dev_run.enable:
    #    val_tasks_split = train_tasks_split = [[0, 1], [2, 3], [4, 5]]

    check(split=train_tasks_split, 
        num_classes=args.model.num_classes, 
        num_tasks=args.config.num_tasks,
        enable_robust=args.model.robust.enable,
    )

    model = set_manager.model_overlay(
        model=source_model,
        loss_f=clModel_default_loss_f,
        data_passer=data_passer,
        args=args,
        args_map={
            'onehot.means': one_hot_means,
        },
        cfg_map={
            'cfg_layer_replace': CLModel.LayerReplace(
                enable=args.model.layer_replace.enable,
                source=torch.nn.ReLU,
                destination_f=lambda a, b, x: GaussA(30),
            ) 
        },
    )
    print(f'MODEL TYPE: {model.get_obj_str_type()}')

    #from lucent.modelzoo.util import get_model_layers
    #print(get_model_layers(model))
    #exit()

    #h = utils.get_model_hierarchy(model)
    #print(h)
    #exit()
    

    cl_data_module = CLDataModule(
        data_transform=train_data_transform,
        dataset_class=dataset_class,
        select_dream_tasks_f=select_dream_tasks_f,
        dream_image_f=dream_image_f,
        render_transforms=render_transforms,
        fast_dev_run=args.fast_dev_run.enable,
        fast_dev_run_dream_threshold=args.fast_dev_run.vis_threshold,
        dream_objective_f=objective_f,
        empty_dream_dataset=dream_sets.DreamDataset(enable_robust=args.model.robust.enable, transform=dreams_transforms),
        progress_bar=progress_bar,
        target_processing_f=target_processing_f,
        logger=logger,
        datasampler=datasampler,
        data_passer=data_passer,
        args=args,
        args_map={
            'dataset_labels': dataset_class_labels,
            'train_tasks_split': train_tasks_split,
            'val_tasks_split': val_tasks_split
        },
    )

    if(args.wandb.watch.enable):
        logger.watch(model, log='all', log_freq=args.wandb.log_freq)
        #wandb.watch(model, log_freq=args.wandb.watch.log_freq, log_graph=args.wandb.watch.log_graph)
    

    trainer = pl.Trainer(
        max_epochs=-1,  # This value doesn't matter
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run.batches if args.fast_dev_run.enable else False, # error when multiple tasks - in new task 0 batches are done.
        limit_train_batches=args.fast_dev_run.batches if args.fast_dev_run.enable else None,
        gpus=None if args.config.cpu else "0,",
        log_every_n_steps=1 if args.fast_dev_run.enable else 50,
        num_sanity_val_steps=num_sanity_val_steps,
    )
    progress_bar._init_progress(trainer)    

    print(f"Fast dev run is {args.fast_dev_run.enable}")
    
    internal_fit_loop = trainer.fit_loop
    custom_loop = CLLoop(
        plan=[args.loop.schedule] * args.config.num_tasks,
        args=args,
        fast_dev_run_epochs=args.fast_dev_run.epochs,
        fast_dev_run=args.fast_dev_run.enable,
        data_passer=data_passer,
        data_module=cl_data_module,
        progress_bar=progress_bar,
    )
    trainer.fit_loop = custom_loop
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule=cl_data_module)

    trainer.test(model, datamodule=cl_data_module)
    if(args.wandb.watch.enable):
        logger.experiment.unwatch(model)
    cl_data_module.flush_wandb()
    if(can_export_config(args)):
        export_config(args, custom_loop.save_folder)

    if(args.loop.layer_stats.use_at is not None):
        plot_pca_graph(custom_loop.model_stats, model=model, overestimated_rank=args.pca_estimate_rank)
        plot_std_stats_graph(model_stats=custom_loop.model_stats, model=model, filepath=custom_loop.save_folder)

    collect_model_information(
        args=args,
        model=model, 
        attack_kwargs=args.model.robust, 
        dataset_class=dataset_class, 
        train_tasks_split=train_tasks_split, 
        collect_main_split=collect_main_split, 
        logger=logger, 
        dreams_transforms=dreams_transforms, 
        set_manager=set_manager,
        custom_loop=custom_loop
    )

def collect_model_information(args, model, attack_kwargs, dataset_class, train_tasks_split, collect_main_split, logger, dreams_transforms, set_manager, custom_loop):
    if(not args.stat.compare_latent and not args.stat.disorder_dream and not args.stat.collect_stats):
        return
    collect_numb_of_points = 2500
    nrows = 4
    ncols = 4
    compare_latent_step_sample = False
    test_sigma_disorder = 0.0
    start_img_value = 0.0

    transform = transforms.Compose([transforms.ToTensor()])
    if(args.stat.collect_stats and not args.fast_def_run.enable):
        print('STATISTICS: Collecting model stats')
        dataset = dataset_class(root="./data", train=False, transform=transform)
        dataset = CLDataModule._split_dataset(dataset, [np.concatenate(train_tasks_split, axis=0)])[0]
        targets = None
        if isinstance(dataset, torch.utils.data.Subset):
            targets = np.take(dataset.dataset.targets, dataset.indices).tolist()
        else:
            targets = dataset.targets
        collector_batch_sampler = PairingBatchSamplerV2(
                    dataset=dataset,
                    batch_size=args.datamodule.batch_size,
                    shuffle=True,
                    classes=np.unique(targets),
                    main_class_split=collect_main_split,
                )
        collect_stats(model=model, dataset=dataset,
            collect_numb_of_points=collect_numb_of_points, 
            collector_batch_sampler=collector_batch_sampler,
            nrows=nrows, ncols=ncols, 
            logger=logger, attack_kwargs=attack_kwargs, path=custom_loop.save_folder)
    if(args.stat.compare_latent and not args.fast_def_run.enable):
        print('STATISTICS: Compare latent')
        compare_latent = CompareLatent()
        compare_latent(
            model=model,
            #loss_f=model.loss_f, 
            used_class=0, 
            logger=logger,
            dream_fetch_transform=dreams_transforms,
            target_processing_f=set_manager.target_processing if set_manager.is_target_processing_latent() else None,
            loss_obj_step_sample=compare_latent_step_sample,
            enable_transforms=not args.datamodule.vis.disable_transforms,
        )
    if(args.stat.disorder_dream and not args.fast_def_run.enable):
        print('STATISTICS: Disorder dream')
        disorder_dream = DisorderDream()
        dataset = dataset_class(root="./data", train=True, transform=transform)
        disorder_dream(
            model=model,
            used_class=1, 
            logger=logger,
            dataset=dataset,
            dream_transform=dreams_transforms,
            #objective_f=set_manager.dream_objective if set_manager.is_target_processing_latent() else None,
            sigma_disorder=test_sigma_disorder,
            start_img_value=start_img_value,
        )

    # show dream png
    #cl_data_module.dreams_dataset.dreams[-1].show()

def extract_data_from_key(data:dict) -> dict:
    """
        Get dict like [(torch_size, num_classes)] and extract it as [torch_size][num_classes]
    """
    new = dict()
    for k, v in data.items():
        if(new.get(k[0]) is None):
            new[k[0]] = dict()
        new[k[0]][k[1]] = v
    return new

def plot_pca_graph(model_stats:dict, model:torch.nn.Module, overestimated_rank:int, filepath:str=None):
    out = pca(model_stats, overestimated_rank=overestimated_rank)
    plotter = PointPlot()
    to_plot = dict()
    path = filepath if filepath is not None else Path(model.name()) / "pca"
    for layer_name, v in out.items():
        to_plot[layer_name] = extract_data_from_key(data=v)
    for layer_name, v in to_plot.items(): 
        for torch_size, vv in v.items():
            plotter.plot_bar(vv, name= path / layer_name / torch_size ,nrows=int(np.sqrt(len(vv))), ncols=int(np.sqrt(len(vv))) + 1)
            

def plot_std_stats_graph(model_stats:dict, model:torch.nn.Module, filepath:str):
    plotter = PointPlot()
    path = filepath if filepath is not None else Path(model.name())
    for layer_name, v in model_stats.items(): 
        v = v.get_const_data()
        std = v.std
        std = extract_data_from_key(data=std)
        for torch_size, vv in std.items(): 
            plotter.plot_errorbar(vv, name=path / "pca" / "std" / layer_name / torch_size)

def collect_stats(model, dataset, collect_numb_of_points, collector_batch_sampler, attack_kwargs, path, nrows=1, ncols=1, logger=None):
    stats = Statistics()
    dataloader = DataLoader(dataset, 
        num_workers=4, 
        pin_memory=False,
        batch_sampler=collector_batch_sampler
    )

    if(attack_kwargs.enable):
        def invoker(model, input):
            xe = model.forward(
                input,
                **vars(attack_kwargs.kwargs)
            )
            if(isinstance(xe, tuple)):
                xe = xe[0]
            return xe
    else:
        def invoker(model, input):
            xe = model.forward(input)
            if(isinstance(xe, tuple)):
                xe = xe[0]
            return xe
        
    buffer = stats.collect(model=model, dataloader=dataloader, num_of_points=collect_numb_of_points, to_invoke=invoker,
        logger=logger)
    plotter = PointPlot()

    if(isinstance(path, str)):
        path = Path(path)

    #plotter.plot(buffer, plot_type='singular', name='plots/singular', show=False, symetric=False, markersize=3, ftype='png')
    #std_mean_dict = Statistics.by_class_operation(Statistics.f_mean_std, buffer, 'saves/mean_std.txt')
    std_mean_distance_dict = Statistics.by_class_operation(Statistics.f_distance, buffer, path / 'saves/distance.txt')
    std_mean_distance_dict = Statistics.by_class_operation(
        Statistics.f_average_point_dist_from_means, 
        buffer, 
        path / 'saves/average_point_dist_from_means.txt', 
        output=std_mean_distance_dict
    )
    Statistics.mean_distance(std_mean_distance_dict)
    plotter.plot_3d(buffer, std_mean_distance_dict, name=path / 'plots/point-plot-3d')
    plotter.plot_std_mean(std_mean_distance_dict, name=path / 'plots/std-mean', show=False, ftype='png')
    plotter.plot_distance(std_mean_distance_dict, nrows=nrows, ncols=ncols, name=path / 'plots/distance_class', show=False, ftype='png', markersize=3)
    plotter.plot_mean_distance(std_mean_distance_dict, name=path / 'plots/mean_distance', show=False, ftype='png', markersize=4)
    plotter.plot_mean_dist_matrix(std_mean_distance_dict, name=path / 'plots/mean_dist_matrix', show=False)
    plotter.saveBuffer(buffer, name=path / 'saves/latent')

def check(split, num_classes, num_tasks, enable_robust):
    test = set()
    for s in split:
        test = test.union(s)
    if(len(test) != num_classes):
        raise Exception(f"Wrong number of classes: {num_classes} / train or validation split: {len(test)}.")
    if(len(split) != num_tasks):
        raise Exception(f"Wrong number of tasks: {num_tasks} / train or validation split size: {len(split)}.")

if __name__ == "__main__":
    args, _ = arg_parser()
    logic(args)