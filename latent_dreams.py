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
from utils import pretty_print as pp
from utils.data_collector import collect_data
from tests.evaluation.single_dream import SingleDream

def data_transform():
    return transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
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
        pp.sprint(f'{pp.COLOR.NORMAL}INFO: Selected None datasampler')
        return None
    elif(dtype == 'v2'):
        pp.sprint(f'{pp.COLOR.NORMAL}INFO: Selected PairingBatchSamplerV2 datasampler')
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
    dreams_transforms = data_transform()

    datasampler = select_datasampler(dtype=args.config.datasampler_type, main_split=main_split)

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
    logger = WandbLogger(project="continual_dreaming", tags=tags, offline=wandb_offline, mode=wandb_mode, name=wandb_run_name(args, wandb_run_id),
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
    pp.sprint(f"{pp.COLOR.NORMAL}Selected configuration:{pp.COLOR.RESET}\n{str(set_manager)}")

    model_latent_size = args.model.num_classes if args.model.latent.size is None else args.model.latent.size
    source_model = set_manager.model(default_weights=args.model.default_weights, num_classes=model_latent_size)
    target_processing_f = set_manager.target_processing
    select_dream_tasks_f = set_manager.select_task
    objective_f = set_manager.dream_objective
    val_tasks_split = train_tasks_split = set_manager.task_split(
        args.config.split.num_classes if args.config.split.num_classes is not None else args.model.num_classes, 
        args.config.num_tasks
    )

    #model_summary(source_model)

    if(args.model.robust.enable):
        print('WARNING:\tTRAIN ROBUSTLY IS ENABLED, SLOWER TRAINING.')

    if(args.fast_dev_run.enable):
        args.datamodule.vis.per_target = 64

    dataset_class, dataset_class_labels = getDataset(args.config.dataset)

    #if args.fast_dev_run.enable:
    #    val_tasks_split = train_tasks_split = [[0, 1], [2, 3], [4, 5]]

    #check(split=train_tasks_split, 
    #    num_classes=args.model.num_classes, 
    #    num_tasks=args.config.num_tasks,
    #    enable_robust=args.model.robust.enable,
    #)

    model = set_manager.model_overlay(
        model=source_model,
        loss_f=clModel_default_loss_f,
        data_passer=data_passer,
        args=args,
        cfg_map={
            'cfg_layer_replace': CLModel.LayerReplace(
                enable=args.model.layer_replace.enable,
                source=torch.nn.ReLU,
                destination_f=lambda a, b, x: GaussA(30),
            ) 
        },
    )
    pp.sprint(f'{pp.COLOR.NORMAL_3}MODEL TYPE: {model.get_obj_str_type()}')

    #from lucent.modelzoo.util import get_model_layers
    #print(get_model_layers(model))
    #exit()

    #h = utils.get_model_hierarchy(model)
    #print(h)
    #exit()
    
    test_transforms = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
    )

    cl_data_module = CLDataModule(
        data_transform=data_transform(),
        test_transform=test_transforms,
        dataset_class=dataset_class,
        select_dream_tasks_f=select_dream_tasks_f,
        dream_image_f=dream_image_f,
        render_transforms=render_transforms,
        fast_dev_run=args.fast_dev_run.enable,
        fast_dev_run_dream_threshold=args.fast_dev_run.vis_threshold,
        dream_objective_f=objective_f,
        empty_dream_dataset=dream_sets.DreamDataset(transform=test_transforms),
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

    pp.sprint(f"{pp.COLOR.WARNING}Fast dev run is {args.fast_dev_run.enable}")
    
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

    if(not args.config.test.disable):
        trainer.test(model, datamodule=cl_data_module)
    if(args.wandb.watch.enable):
        logger.experiment.unwatch(model)
    cl_data_module.flush_wandb()
    if(not args.fast_dev_run.enable and not custom_loop.cfg_save.ignore_config):
        export_config(args, custom_loop.save_folder, 'run_config.json')

    if(utils.check_python_enabled(args.loop.layer_stats.use_at)):
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
        custom_loop=custom_loop,
        sigma_disorder=args.stat.disorder.sigma,
        start_img_value=args.stat.disorder.start_img_val,
        try_except=False,
        multitarget=args.datamodule.vis.multitarget.enable,
        num_classes=args.model.num_classes,
    )

def collect_model_information(args, model, attack_kwargs, dataset_class, train_tasks_split, 
                              collect_main_split, logger, dreams_transforms, set_manager, custom_loop,
                              sigma_disorder, start_img_value, num_classes,
                              try_except=True, multitarget=False):
    collect_numb_of_points = 2500
    nrows = 4
    ncols = 4
    compare_latent_step_sample = False

    def run(f, name):
        if(try_except):
            try:
                f()
            except Exception as e_name:
                pp.sprint(f"{pp.COLOR.WARNING}ERROR: {name} could not be completed. Error:\n{e_name}")
        else:
            f()

    transform = transforms.Compose([transforms.ToTensor()])
    if(args.stat.collect_stats and not args.fast_dev_run.enable):
        pp.sprint(f'{pp.COLOR.NORMAL}STATISTICS: Collecting model stats')
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
    if(args.stat.compare_latent and not args.fast_dev_run.enable):
        pp.sprint(f'{pp.COLOR.NORMAL}STATISTICS: Compare latent')
        compare_latent = CompareLatent()
        compare_latent_call = lambda: compare_latent(
            model=model,
            #loss_f=model.loss_f, 
            used_class=0, 
            logger=logger,
            dream_fetch_transform=dreams_transforms,
            target_processing_f=set_manager.target_processing if set_manager.is_target_processing_latent() else None,
            loss_obj_step_sample=compare_latent_step_sample,
            enable_transforms=not args.datamodule.vis.disable_transforms,
        )
        run(compare_latent_call, 'compare latent')
    if(args.stat.disorder_dream and not args.fast_dev_run.enable):
        pp.sprint(f'{pp.COLOR.NORMAL}STATISTICS: Disorder dream')
        disorder_dream = DisorderDream()
        dataset = dataset_class(root="./data", train=True, transform=transform)
        disorder_dream_call = lambda: disorder_dream(
            model=model,
            used_class=1, 
            logger=logger,
            dataset=dataset,
            dream_transform=dreams_transforms,
            #objective_f=set_manager.dream_objective if set_manager.is_target_processing_latent() else None,
            sigma_disorder=sigma_disorder,
            start_img_value=start_img_value,
        )
        run(disorder_dream_call, 'disorder dreams')
    if(args.stat.collect.latent_buffer.enable and not args.fast_dev_run.enable):
        if(hasattr(model.loss_f, 'cloud_data')):
            namepath = args.stat.collect.latent_buffer.name
            if(namepath is None):
                namepath = custom_loop.save_folder / 'default.csv'
            pp.sprint(f'{pp.COLOR.NORMAL}STATISTICS: collect points from latent buffer')
            def sample_latent_buffer_call():
                sample_latent_buffer_mean_std(
                    buffer=model.loss_f.cloud_data,
                    namepath=namepath,
                    cl_idx=args.stat.collect.latent_buffer.cl_idx,
                )
                cl_idx = args.stat.collect.latent_buffer.cl_idx
                cl_idx = cl_idx if cl_idx is not None else range(args.model.num_classes)
                sample_latent_buffer_target_process(
                    model=model,
                    target_process_f=set_manager.target_processing,
                    namepath=namepath,
                    size=args.stat.collect.latent_buffer.size,
                    cl_idx=cl_idx,
                    multitarget=multitarget,
                )
            run(sample_latent_buffer_call, 'collect latent buffer points')
        else:
            pp.sprint(f'{pp.COLOR.WARNING}STATISTICS WARNING: model`s loss function does not have latent buffer')
    if(args.stat.collect.single_dream.enable and not args.fast_dev_run.enable):
        tmp = SingleDream(
            drift_sigma=args.stat.collect.single_dream.sigma,
            logger=logger,
            num_classes=num_classes,
        )
        single_dream_call = lambda: tmp(
            model=model,
        )
        run(single_dream_call, 'single dream')
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
    try:
        out = pca(model_stats, overestimated_rank=overestimated_rank)
        if(out is None):
            return
        plotter = PointPlot()
        to_plot = dict()
        path = filepath if filepath is not None else Path(model.name()) / "pca"
        for layer_name, v in out.items():
            to_plot[layer_name] = extract_data_from_key(data=v)
        for layer_name, v in to_plot.items(): 
            for torch_size, vv in v.items():
                plotter.plot_bar(vv, name= path / layer_name / str(torch_size) ,nrows=int(np.sqrt(len(vv))), ncols=int(np.sqrt(len(vv))) + 1)
    except Exception as pca_exception:
        pp.sprint(f'{pp.COLOR.WARNING}WARNING: PCA graph plot failed. Exception:\n{pca_exception}')
            
def plot_std_stats_graph(model_stats:dict, model:torch.nn.Module, filepath:str):
    try:
        plotter = PointPlot()
        path = filepath if filepath is not None else Path(model.name())
        for layer_name, v in model_stats.items(): 
            v = v.get_const_data()
            std = v.std
            if(std is None):
                pp.sprint(f"{pp.COLOR.WARNING}WARNING: std graph cannot be plotted.")
                return
            std = extract_data_from_key(data=std)
            for torch_size, vv in std.items(): 
                plotter.plot_errorbar(vv, name=path / "pca" / "std" / layer_name / str(torch_size))
    except Exception as std_stat_exception:
        pp.sprint(f'{pp.COLOR.WARNING}WARNING: std stat graph plot failed. Exception:\n{std_stat_exception}')

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

def sample_latent_buffer_mean_std(buffer, namepath, cl_idx=None):
    if(not isinstance(namepath, Path)):
        namepath = Path(namepath)

    if(cl_idx is None):
        collect_mean = lambda current_cl: lambda idx: buffer.mean_target(current_cl)
        collect_std = lambda current_cl: lambda idx: buffer.std_target(current_cl)
        data = utils.dict_to_tensor(buffer.mean())
        for cl in range(data.shape[0]):
            mean_name = namepath.parent / (f'mean.' + str(namepath.name))
            std_name = namepath.parent / (f'std.' + str(namepath.name))
            header = [f'cl_{cl}-dim{y}' for y in range(data.shape[-1])]
            collect_data(collect_f=collect_mean(cl), size=1, namepath=mean_name, mode='a', header=header)
            collect_data(collect_f=collect_std(cl), size=1, namepath=std_name, mode='a', header=header)
    else:
        collect_mean = lambda idx: buffer.mean_target(cl_idx)
        collect_std = lambda idx: buffer.std_target(cl_idx)
        d_size = collect_mean(0).shape[-1]
        header = [f'cl_{int(cl_idx)}-dim_{x}' for x in range(d_size)]
        mean_name = namepath.parent / ('mean.' + str(namepath.name))
        std_name = namepath.parent / ('std.' + str(namepath.name))
        collect_data(collect_f=collect_mean, size=1, namepath=mean_name, mode='a', header=header)
        collect_data(collect_f=collect_std, size=1, namepath=std_name, mode='a', header=header)

def sample_latent_buffer_target_process(model, target_process_f, namepath, size, cl_idx:torch.Tensor, multitarget):    
    def run_singletarget(cl_idx, namepath):
        collect_point_call = lambda idx: target_process_f(target=cl_idx, model=model)
        header = [f'cl_{cl_idx}-dim_{x}' for x in range(collect_point_call(0).shape[0])]
        if(not isinstance(namepath, Path)):
            namepath = Path(namepath)
        name = namepath.parent / 'sample' / (f'cl_{cl_idx}.' + str(namepath.name))
        collect_data(collect_f=collect_point_call, size=size, namepath=name, mode='a', header=header)

    def run_multitarget(cl_idx, namepath):
        collect_point_call = lambda idx: target_process_f(target=cl_idx, model=model).flatten()
        #cl = ''
        #for c in cl_idx.numpy():
        #    cl = cl + f'{c}-'
        #cl = cl[: -1]
        header = [f'cl_{c}-dim_{x}' for c in cl_idx.numpy() for x in range(target_process_f(target=cl_idx, model=model).shape[0])]
        if(not isinstance(namepath, Path)):
            namepath = Path(namepath)
        name = namepath.parent / (f'sample_cl_full.' + str(namepath.name))
        collect_data(collect_f=collect_point_call, size=size, namepath=name, mode='a', header=header)

    if(isinstance(cl_idx, int)):
        cl_idx = torch.tensor(cl_idx)
        run_singletarget(cl_idx, namepath=namepath)
    elif(isinstance(cl_idx, list) or multitarget):
        cl_idx = torch.tensor(cl_idx)
        run_multitarget(cl_idx, namepath=namepath)
    else:
        for cl in cl_idx:
            run_singletarget(torch.tensor(cl), namepath=namepath)
    



if __name__ == "__main__":
    args, _ = arg_parser()
    logic(args)