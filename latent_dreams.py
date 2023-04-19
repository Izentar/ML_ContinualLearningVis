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
from my_parser import arg_parser, log_to_wandb, attack_args_to_kwargs, optim_params_to_kwargs, wandb_run_name

from utils.load import try_load
from config.default import robust_data_path
from model.statistics.base import pca
from collections.abc import Sequence
from utils.utils import parse_image_size
import wandb
from dream.image import Image
from utils import utils
from model.activation_layer import GaussA

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

    
    if(ptype == 'fft' or ptype == 'pixel'):
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
        return None
    elif(dtype == 'v2'):
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
    if(args.seed is not None):
        pl.seed_everything(args.seed)

    main_split = collect_main_split = 0.5
    sigma = 0.01
    rho = 1.
    wandb_offline = False if not args.fast_dev_run else True
    wandb_mode = "online" if not args.fast_dev_run else "disabled"
    #wandb_offline = True

    optimizer = lambda param: torch.optim.Adam(param, lr=args.lr)
    #optimizer = lambda param: torch.optim.SGD(param, lr=1e-9, momentum=0.1, weight_decay=0.1)
    #scheduler = None
    scheduler = lambda optim: torch.optim.lr_scheduler.ExponentialLR(optim, gamma=0.1)
    def reset_optim(optim, sched=None):
        for g in optim.param_groups:
            g['lr'] = args.lr

    layer_stats_hook_to:list[str]|None=args.layer_stats_hook_to
    layer_stats_verbose=False
    layer_stats_flush_to_disk=False
    layer_stats_loss_device='cuda:0'
    layer_stats_collect_device='cuda:0'
    
    num_sanity_val_steps = 0
    dream_optim = lambda params: torch.optim.Adam(params, lr=args.dream_lr)
    train_data_transform = data_transform() #transforms.Compose([transforms.ToTensor()])
    dreams_transforms = data_transform()

    datasampler = select_datasampler(dtype=args.datasampler_type, main_split=main_split)

    attack_kwargs = attack_args_to_kwargs(args)
    optim_params = optim_params_to_kwargs(args)

    one_hot_means = get_one_hots(mytype='diagonal', size=args.number_of_classes)
    clModel_default_loss_f = torch.nn.CrossEntropyLoss()
    dream_image_f = param_f_create(ptype=args.dream_image_type)
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
    if args.fast_dev_run:
        tags = ["fast_dev_run"]
    logger = WandbLogger(project="continual_dreaming", tags=tags, offline=wandb_offline, mode=wandb_mode, name=wandb_run_name(args))
    if(log_args_to_wandb):
        log_to_wandb(args)
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

    #### setting in dream_obj_type a diversity type will extend time of dreaming significantly
    set_manager = FunConfigSetPredefined(
        name_type=args.framework_type, 
        mtype=args.model_type,
        select_task_type=args.select_task_type,
        target_processing_type=args.target_processing_type,
        task_split_type=args.task_split_type,
        otype=args.overlay_type,
        #mtype='SAEGAUSS', 
        #mtype='RESNET18', 
        dream_obj_type=args.dream_obj_type,
        #dream_obj_type=["objective-channel", "OBJECTIVE-DLA-DIVERSITY-1", "OBJECTIVE-DLA-DIVERSITY-2", "OBJECTIVE-DLA-DIVERSITY-3"],
        logger=logger
    )
    set_manager.init_dream_objectives(logger=logger, label='dream')
    print(f"Selected configuration:\n{str(set_manager)}")

    source_model = set_manager.model(num_classes=args.number_of_classes)
    target_processing_f = set_manager.target_processing
    select_dream_tasks_f = set_manager.select_task
    objective_f = set_manager.dream_objective
    val_tasks_split = train_tasks_split = set_manager.task_split(args.number_of_classes, args.num_tasks)

    #model_summary(source_model)

    if(args.enable_robust):
        print('WARNING:\tTRAIN ROBUSTLY IS ENABLED, SLOWER TRAINING.')

    if(args.fast_dev_run):
        args.dreams_per_target = 64

    dream_dataset_class = dream_sets.DreamDatasetWithLogits if args.train_with_logits else dream_sets.DreamDataset
    dataset_class, dataset_class_labels = getDataset(args.dataset)

    #if args.fast_dev_run:
    #    val_tasks_split = train_tasks_split = [[0, 1], [2, 3], [4, 5]]

    check(split=train_tasks_split, 
        num_classes=args.number_of_classes, 
        num_tasks=args.num_tasks,
        enable_robust=args.enable_robust,
    )

    model = try_load(args.export_path, args.load_model)

    if(model is None):
        model = set_manager.model_overlay(
            model=source_model,
            load_model=args.load_model,
            robust_dataset_name=args.dataset,
            robust_data_path=robust_data_path,
            num_tasks=args.num_tasks,
            num_classes=args.number_of_classes,
            attack_kwargs=attack_kwargs,
            dreams_with_logits=args.train_with_logits,
            dream_frequency=args.dream_frequency,
            sigma=sigma,
            rho=rho,
            norm_lambda=args.norm_lambda,
            hidden=args.number_of_classes,
            one_hot_means=one_hot_means,
            cyclic_latent_buffer_size_per_class=args.cyclic_latent_buffer_size_per_class,
            loss_f=clModel_default_loss_f,
            data_passer=data_passer,
            optimizer_construct_type=args.optimizer_type,
            scheduler_construct_type=args.scheduler_type,
            scheduler_steps=args.train_scheduler_steps,
            optimizer_restart_params_type=args.reset_optim_type,
            optimizer_params=optim_params,
            replace_layer=args.replace_layer,
            replace_layer_from=torch.nn.ReLU,
            replace_layer_to_f=lambda a, b, x: GaussA(30),
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
        train_tasks_split=train_tasks_split,
        dataset_class=dataset_class,
        dreams_per_target=args.dreams_per_target,
        val_tasks_split=val_tasks_split,
        select_dream_tasks_f=select_dream_tasks_f,
        dream_image_f=dream_image_f,
        dream_image_type=args.dream_image_type,
        render_transforms=render_transforms,
        fast_dev_run=args.fast_dev_run,
        fast_dev_run_dream_threshold=args.fast_dev_run_dream_threshold,
        dream_threshold=args.dream_threshold,
        dream_objective_f=objective_f,
        empty_dream_dataset=dream_dataset_class(enable_robust=args.enable_robust, transform=dreams_transforms),
        progress_bar=progress_bar,
        target_processing_f=target_processing_f,
        dreaming_batch_size=args.dreaming_batch_size,
        optimizer=dream_optim,
        logger=logger,
        dataset_class_labels=dataset_class_labels,
        datasampler=datasampler,
        batch_size=args.batch_size,
        enable_dream_transforms=not args.disable_dream_transforms,
        shuffle=args.disable_shuffle,
        dream_shuffle=args.disable_dream_shuffle,
        num_workers=args.num_workers,
        dream_num_workers=args.dream_num_workers,
        test_val_num_workers=args.test_val_num_workers,
        train_only_dream_batch_at=args.train_only_dream_batch_at,
        standard_image_size=args.standard_image_size,
        data_passer=data_passer,
        dream_decorrelate=args.decorrelate,
    )
    

    trainer = pl.Trainer(
        max_epochs=-1,  # This value doesn't matter
        logger=logger,
        callbacks=callbacks,
        fast_dev_run=args.fast_dev_run_batches if args.fast_dev_run else False, # error when multiple tasks - in new task 0 batches are done.
        limit_train_batches=args.fast_dev_run_batches if args.fast_dev_run else None,
        gpus=None if args.cpu else "0,",
        log_every_n_steps=1 if args.fast_dev_run else 50,
        num_sanity_val_steps=num_sanity_val_steps,
    )
    progress_bar._init_progress(trainer)

    #WandbLogger.log("model/sigmaParams": )
    

    print(f"Fast dev run is {args.fast_dev_run}")

    
    internal_fit_loop = trainer.fit_loop
    custom_loop = CLLoop(
        [args.epochs_per_task] * args.num_tasks, 
        enable_dreams_gen_at=args.enable_dreams_gen_at,
        fast_dev_run_epochs=args.fast_dev_run_epochs,
        fast_dev_run=args.fast_dev_run,
        data_passer=data_passer,
        num_loops=args.num_loops,
        run_training_at=args.run_training_at,
        early_finish_at=args.early_finish_at,
        reload_model_at=args.reload_model_at,
        reinit_model_at=args.reinit_model_at,
        weight_reset_sanity_check=args.weight_reset_sanity_check,
        enable_checkpoint=args.enable_checkpoint,
        save_model_inner_path=args.save_model_inner_path,
        save_trained_model=args.save_trained_model,
        load_model=args.load_model,
        export_path=args.export_path,
        save_dreams=args.save_dreams,
        load_dreams=args.load_dreams,
        gather_layer_loss_at=args.gather_layer_loss_at,
        use_layer_loss_at=args.use_layer_loss_at,
        data_module=cl_data_module,
        progress_bar=progress_bar,
        layer_stats_hook_to=layer_stats_hook_to,
        layer_stats_verbose=layer_stats_verbose,
        layer_stats_flush_to_disk=layer_stats_flush_to_disk,
        layer_stats_loss_device=layer_stats_loss_device,
        layer_stats_collect_device=layer_stats_collect_device,
        advance_clear_dreams=args.advance_clear_dreams,
        save_layer_stats=args.save_layer_stats,
        load_layer_stats=args.load_layer_stats,
        ll_scaling=args.ll_scaling,
        use_grad_pruning_at=args.use_grad_pruning_at,
        grad_pruning_percent=args.grad_pruning_percent,
        use_grad_activ_pruning_at=args.use_grad_activ_pruning_at,
        grad_activ_pruning_percent=args.grad_activ_pruning_percent,
        layer_loss_del_cov_after=args.ll_del_cov_after,
    )
    trainer.fit_loop = custom_loop
    trainer.fit_loop.connect(internal_fit_loop)
    trainer.fit(model, datamodule=cl_data_module)

    trainer.test(model, datamodule=cl_data_module)
    cl_data_module.flush_wandb()

    if(args.gather_layer_loss_at is not None):
        plot_pca_graph(custom_loop.model_stats, model=model, overestimated_rank=args.pca_estimate_rank)
        plot_std_stats_graph(model_stats=custom_loop.model_stats, model=model)

    collect_model_information(
        args=args,
        model=model, 
        attack_kwargs=attack_kwargs, 
        dataset_class=dataset_class, 
        train_tasks_split=train_tasks_split, 
        collect_main_split=collect_main_split, 
        logger=logger, 
        dreams_transforms=dreams_transforms, 
        set_manager=set_manager,
    )

def collect_model_information(args, model, attack_kwargs, dataset_class, train_tasks_split, collect_main_split, logger, dreams_transforms, set_manager):
    if(not args.compare_latent and not args.disorder_dream and not args.collect_stats):
        return
    collect_numb_of_points = 2500
    nrows = 4
    ncols = 4
    compare_latent_step_sample = False
    test_sigma_disorder = 0.0
    start_img_value = 0.0

    transform = transforms.Compose([transforms.ToTensor()])
    if(args.collect_stats):
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
                    batch_size=args.batch_size,
                    shuffle=True,
                    classes=np.unique(targets),
                    main_class_split=collect_main_split,
                )
        collect_stats(model=model, dataset=dataset,
            collect_numb_of_points=collect_numb_of_points, 
            collector_batch_sampler=collector_batch_sampler,
            nrows=nrows, ncols=ncols, 
            logger=logger, attack_kwargs=attack_kwargs)
    if(args.compare_latent):
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
            enable_transforms=not args.disable_dream_transforms,
        )
    if(args.disorder_dream):
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
        Get dict like [(torch_size, class_number)] and extract it as [torch_size][class_number]
    """
    new = dict()
    for k, v in data.items():
        if(new.get(k[0]) is None):
            new[k[0]] = dict()
        new[k[0]][k[1]] = v
    return new

def plot_pca_graph(model_stats:dict, model:torch.nn.Module, overestimated_rank:int):
    out = pca(model_stats, overestimated_rank=overestimated_rank)
    plotter = PointPlot()
    to_plot = dict()
    for layer_name, v in out.items():
        to_plot[layer_name] = extract_data_from_key(data=v)
    for layer_name, v in to_plot.items(): 
        for torch_size, vv in v.items():
            plotter.plot_bar(vv, name=f'{model.name()}/pca/{layer_name}/{torch_size}',nrows=int(np.sqrt(len(vv))), ncols=int(np.sqrt(len(vv))) + 1)
            

def plot_std_stats_graph(model_stats:dict, model:torch.nn.Module):
    plotter = PointPlot()
    for layer_name, v in model_stats.items(): 
        v = v.get_const_data()
        std = v.std
        std = extract_data_from_key(data=std)
        for torch_size, vv in std.items(): 
            plotter.plot_errorbar(vv, name=f'{model.name()}/pca/std/{layer_name}/{torch_size}')

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