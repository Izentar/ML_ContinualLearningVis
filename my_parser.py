from argparse import ArgumentParser, Namespace
from collections.abc import Callable
from typing import Any
import wandb
import sys
import json
from pathlib import Path
import glob
import os
from utils import utils
import numpy as np
from json import JSONEncoder, JSONDecoder
from utils import pretty_print as pp
from collections.abc import Sequence

def main_arg_parser() -> ArgumentParser:
    parser = ArgumentParser(prog='Continual dreaming', add_help=True, description='Configurable framework to work with\
continual learning tasks.', 
epilog="""
If data is divided into tasks then:
* Validation dataset uses data from test dataset but divided into appropriate tasks.
* Test dataset uses test data with only the classes used in previous tasks. 
""")

    ######################################
    #####       main arguments      ######
    ######################################

    parser.add_argument("--wandb.run.folder", type=str, default="log_run/", help='') ##**
    parser.add_argument("--config.seed", type=int, help='Seed of the pytorch random number generator. Default None - random seed.') ##**
    parser.add_argument("--config.folder", type=str, default='run_conf/', help='Root forlder of the runs.') ##**
    parser.add_argument("--loop.train_at", nargs='+', type=str, default='True', help='Run training at corresponding loop index or indexes. \
If True, run on all loops, if False, do not run. If "loop.gather_layer_loss_at" is set, then it takes precedence.') ##**
    parser.add_argument("--config.load", action="append", help='The config file(s) that should be used. The config file \
takes precedence over command line arguments. Config files will be applied in order of the declaration.')
    parser.add_argument("--config.export", type=str, help='File where to export current config.')
    parser.add_argument("--config.test.disable", action="store_true", help='')
    parser.add_argument("--config.cpu", action="store_true")
    parser.add_argument("-d", "--config.dataset", type=str) ##**
    parser.add_argument("--config.datasampler_type", type=str, default='none', help='''Select datasampler type.
Avaliable types are: 
"none" - no datasampler.
"v2" - datasampler where you can choose division into the majority class and other class in single batch.''')
    parser.add_argument("--model.latent.buffer.size_per_class", type=int, default=40, help='Size per class of the cyclic \
buffer used in island overlay for model.')

    parser.add_argument("--datamodule.disable_shuffle", action="store_true", help='Flag to shuffle train normal and dream datasets. If \
flag "dataloader_disable_dream_shuffle" is set then it takes precedence over this flag.')
    
    # worker numbers, try experimenting with this to speed up computation
    parser.add_argument("--datamodule.num_workers", type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument("--datamodule.vis.num_workers", type=int, help='Number of visualization dataloader workers.')
    parser.add_argument("--datamodule.test_num_workers", type=int, help='Number of test dataloader workers.')
    parser.add_argument("--datamodule.val_num_workers", type=int, help='Number of validation dataloader workers.')

    ######################################
    #####      load/save model      ######
    ######################################
    parser.add_argument("--loop.save.model", action="store_true", help='Save model at the end of all training loops') ##**
    parser.add_argument("--loop.load.model", action="store_true", help='') ##**
    parser.add_argument("--loop.save.enable_checkpoint", action="store_true", help='')
    parser.add_argument("--loop.save.dreams", action="store_true", help='') ##**
    parser.add_argument("--loop.load.dreams", action="store_true", help='') ##**
    parser.add_argument("--loop.save.root", type=str, help='') ##**
    parser.add_argument("--loop.load.root", type=str, help='') ##**
    parser.add_argument("--loop.load.id", nargs='+', type=str, help='') ##**
    parser.add_argument("--loop.load.name", type=str, help='') ##**
    
    ######################################
    #####    numerical parameters   ######
    ######################################
    parser.add_argument("--datamodule.batch_size", type=int, default=32, help='Size of the batch.')
    parser.add_argument("--config.num_tasks", type=int, default=1, help='How many tasks will be created')
    parser.add_argument("--loop.num_loops", type=int, default=1, help='How many loops will be traversed. \
Each new loop will increment current task index. If "num_loops">"num_tasks" then at each new loop \
the last task in array will be used.') ##**
    parser.add_argument("--loop.schedule", nargs='+', type=int, help='How many epochs do per one task in "num_tasks". \
Array size should be the same as num_loops for each loop.') ##**
    parser.add_argument("--model.num_classes", type=int, default=10, help='Number of classes model should output. \
If less than in dataset then model will be trained and validated only using this number of classes')
    parser.add_argument("--model.latent.size", type=int)

    parser.add_argument("--model.optim.type", type=str, default='adam', help='')
    parser.add_argument("--model.optim.reset_type", type=str, default='default', help='')
    parser.add_argument("--model.optim.kwargs.lr", type=float, default=1e-3, help='Learning rate of the optimizer.')
    parser.add_argument("--model.optim.kwargs.gamma", type=float, default=1, help='Gamma parameter for optimizer if exist.')
    parser.add_argument("--model.optim.kwargs.momentum", type=float, default=0, help='')
    parser.add_argument("--model.optim.kwargs.dampening", type=float, default=0, help='')
    parser.add_argument("--model.optim.kwargs.weight_decay", type=float, default=0, help='')
    parser.add_argument("--model.optim.kwargs.betas", nargs='+', type=float, default=[0.9, 0.999], help='')
    parser.add_argument("--model.optim.kwargs.amsgrad", action="store_true", help='')

    parser.add_argument("--model.sched.type", type=str, default='none', help='Type of scheduler. Use "model.scheduler.steps" \
to choose epoch at which to call it.')
    parser.add_argument("--model.sched.kwargs.gamma", default=1., type=float)
    parser.add_argument("--model.sched.kwargs.milestones", nargs='+', type=int)
    parser.add_argument("--model.sched.steps", nargs='+', type=int, default=(3, ), help='Epoch training steps \
at where to call scheduler, change learning rate. Use "model.scheduler.type" to enable scheduler.')

    parser.add_argument("--model.norm_lambda", type=float, default=0., help='Lambda parametr of the used l2 normalization. If 0. then \
no normalization is used. Normalization is used to the last model layer, the latent output of the "CLModelWithIslands".')
    

    ######################################
    #####       configuration       ######
    ######################################
    parser.add_argument("--config.framework_type", type=str, default='sae-chiloss', help='Framework type') ##**
    parser.add_argument("--config.dream_obj_type", nargs='+', type=str, help='Model objective funtions type. May be multiple')
    parser.add_argument("--config.select_task_type", type=str, help='From utils.functional.select_task.py')
    parser.add_argument("--config.target_processing_type", type=str, help='From utils.functional.target_processing.py')
    parser.add_argument("--config.task_split_type", type=str, help='From utils.functional.task_split.py')
    parser.add_argument("--config.overlay_type", type=str, help='Overlay type')
    parser.add_argument("--config.split.num_classes", type=int, help='')
    parser.add_argument("--model.type", type=str, help='Model type') ##**
    parser.add_argument("--model.latent.onehot.type", type=str, default='diagonal', help='Model onehot types') ##**

    parser.add_argument("--model.default_weights", action="store_true", help='') ##**


    ######################################
    #####       model reinit        ######
    ######################################
    parser.add_argument("--loop.model.reload_at", nargs='+', type=str, help='Reload model weights after each main loop.\
Reloading means any weights that model had before training will be reloaded. Reload is done AFTER dream generation if turned on.') ##**
    parser.add_argument("--loop.model.reinit_at", nargs='+', type=str, help='Reset model after each main loop.\
Model will have newly initialized weights after each main loop. Reinit is done AFTER dream generation if turned on.') ##**

    parser.add_argument("--loop.weight_reset_sanity_check", action="store_true", help='Enable sanity check for reload/reinit weights.')
    parser.add_argument("--model.train_sanity_check", action="store_true", help='Enable sanity check for reload/reinit weights.')


    ######################################
    #####       fast dev run        ######
    ######################################
    parser.add_argument("-f", "--fast_dev_run.enable", action="store_true", help='Use to fast check for errors in code.\
It ') ##**
    parser.add_argument("--fast_dev_run.batches", type=int, default=30, help='')
    parser.add_argument("--fast_dev_run.epochs", type=int, default=1, help='')
    parser.add_argument("--fast_dev_run.vis_threshold", nargs='+', type=int, default=[5], help='')


    parser.add_argument("--wandb.watch.enable", action="store_true", help='')
    parser.add_argument("--wandb.watch.log_freq", type=int, default=1000, help='')

    return parser

def robust_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--model.robust.enable", action="store_true", help='Enable robust training.')
    parser.add_argument("--model.robust.data_path", type=str, default="./data", help='')
    parser.add_argument("--model.robust.dataset_name", type=str, help='')
    parser.add_argument("--model.robust.resume_path", type=str, help='')
    parser.add_argument("--model.robust.kwargs.constraint", type=str, default="2", help='')
    parser.add_argument("--model.robust.kwargs.eps", type=float, default=0.5, help='')
    parser.add_argument("--model.robust.kwargs.step_size", type=float, default=1.5, help='')
    parser.add_argument("--model.robust.kwargs.iterations", type=int, default=10, help='')
    parser.add_argument("--model.robust.kwargs.random_start", type=int, default=0, help='')
    parser.add_argument("--model.robust.kwargs.random_restart", type=int, default=0, help='')
    parser.add_argument("--model.robust.kwargs.custom_loss", type=str, help='') # default None
    parser.add_argument("--model.robust.kwargs.use_worst", action="store_false", help='')
    parser.add_argument("--model.robust.kwargs.with_latent", action="store_true", help='')
    parser.add_argument("--model.robust.kwargs.fake_relu", action="store_true", help='')
    parser.add_argument("--model.robust.kwargs.no_relu", action="store_true", help='')
    parser.add_argument("--model.robust.kwargs.make_adversary", action="store_true", help='') # required

    return parser

def layer_statistics_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--loop.layer_stats.use_at", type=str, nargs='+', help='Gather layer statistics \
at given fit loop index. Default None means this functionality is not enabled. Value less than zero \
means it will be used just like the python indexing for negative numbers.') ##**
    parser.add_argument("--loop.vis.layerloss.mean_norm.use_at", type=str, nargs='+', help='Use layer loss function \
at given fit loop index. Default None means this functionality is not enabled. Value less than zero \
means it will be used just like the python indexing for negative numbers.') ##**
    parser.add_argument("--loop.save.layer_stats", action="store_true", help='Path where to save layer_stats.') ##**
    parser.add_argument("--loop.load.layer_stats", action="store_true", help='Path from where to load layer_stats.') ##**
    parser.add_argument("--loop.layer_stats.hook_to", nargs='+', type=str, help='Name of the layers to hook up. If exception \
thrown, list of avaliable layers will be displayed.') ##**
    parser.add_argument("--loop.vis.layerloss.mean_norm.hook_to", nargs='+', type=str)
    parser.add_argument("--loop.vis.layerloss.grad_pruning.hook_to", nargs='+', type=str)
    parser.add_argument("--loop.vis.layerloss.grad_activ_pruning.hook_to", nargs='+', type=str)
    parser.add_argument("--model.layer_replace.enable", action='store_true', help='Replace layer. For now replace ReLu to GaussA.') ##**
    parser.add_argument("--loop.vis.layerloss.mean_norm.scale", type=float, default=0.001, help='Scaling for layer loss.') ##**
    parser.add_argument("--loop.vis.layerloss.grad_pruning.use_at", type=str, nargs='+', help='Use gradient pruning at.') ##**
    parser.add_argument("--loop.vis.layerloss.grad_pruning.percent", type=float, default=0.01,
        help='Percent of gradient pruning neurons at given layer. Selected by std descending.') ##**
    parser.add_argument("--loop.vis.layerloss.grad_activ_pruning.use_at", type=str, nargs='+', help='') ##**
    parser.add_argument("--loop.vis.layerloss.grad_activ_pruning.percent", type=float, default=0.01, help='') ##**
    parser.add_argument("--loop.vis.layerloss.mean_norm.del_cov_after", action="store_true", help='Delete covariance matrix after calculating inverse of covariance.') ##**

    parser.add_argument("--loop.vis.layerloss.mean_norm.device", type=str, default='cuda', help='')
    parser.add_argument("--loop.vis.layerloss.grad_pruning.device", type=str, default='cuda', help='')
    parser.add_argument("--loop.vis.layerloss.grad_activ_pruning.device", type=str, default='cuda', help='')
    parser.add_argument("--loop.layer_stats.device", type=str, default='cuda', help='')
    parser.add_argument("--loop.layer_stats.flush_to_disk", action="store_true", help='')
    parser.add_argument("--loop.layer_stats.type", nargs='+', type=str, help="Possible types: 'mean', 'std', 'cov'. Default ()'mean, 'std')")

    return parser

def dream_parameters_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--loop.vis.generate_at", nargs='+', type=str, default='False', help='At which loop or loops enable dreaming where framework \
should produce dreams and use them during training. Can take one or more indexes and boolean. Default None will not produce any dreams. \
Without running this and running "loop.train_at" will run only test. Use this with "datamodule.vis.only_vis_at" to train only on dream batch.') ##**
    parser.add_argument("--datamodule.vis.per_target", type=int, default=128, help='How many epochs do per one task in "num_tasks"') ##**
    parser.add_argument("--datamodule.vis.multitarget.enable", action="store_true", help='') 
    parser.add_argument("--datamodule.vis.multitarget.random", action="store_true", help='') 
    parser.add_argument("--datamodule.vis.batch_size", type=int, default=128, help='How many images \
in batch during dreaming should be produced.')
    
    parser.add_argument("--datamodule.vis.optim.type", type=str, default='adam', help='')
    parser.add_argument("--datamodule.vis.optim.kwargs.lr", type=float, default=1e-3, help='Learning rate of the dream optimizer.')
    parser.add_argument("--datamodule.vis.optim.kwargs.betas", nargs='+', type=float, default=[0.9, 0.999], help='')
    parser.add_argument("--datamodule.vis.optim.kwargs.gamma", type=float, default=1, help='')
    parser.add_argument("--datamodule.vis.optim.kwargs.weight_decay", type=float, default=0, help='')
    parser.add_argument("--datamodule.vis.optim.kwargs.amsgrad", action="store_true", help='')
    parser.add_argument("--datamodule.vis.optim.kwargs.momentum", type=float, default=0, help='')
    parser.add_argument("--datamodule.vis.optim.kwargs.dampening", type=float, default=0, help='')

    parser.add_argument("--datamodule.vis.sched.type", type=str)
    parser.add_argument("--datamodule.vis.threshold", nargs='+', type=int, default=[512, ], help='How many iterations of the trainig loop should \
be used to generate a batch of images during dreaming, using only max value. Values lesser than max are points where the \
images from the batch will be additionaly saved.')
    parser.add_argument("--datamodule.vis.disable_transforms", action="store_true", help='Enable and add all default \
tranforms on dreamed images used in lucid framework in main function.') ##**
    parser.add_argument("--datamodule.vis.disable_shuffle", action="store_true", help='Flag to shuffle only train dream dataset')
    parser.add_argument("--datamodule.vis.image_type", type=str, default='pixel', help='Type of image. Default \
"fft"; "pixel"; "cppn"(does not use "datamodule.vis.batch_size")')
    parser.add_argument("--datamodule.vis.only_vis_at", nargs='+', type=str, default='False', help='Use this flag to train only on dream batch \
after first epoch when dream batch is created.') ##**
    parser.add_argument("--datamodule.vis.standard_image_size", nargs='+', type=int, help='Tuple of sizes of the image after image transformation during dreaming. \
Checks if the output image has the provided shape. Do not include batch here. Default None.') ##**
    parser.add_argument("--loop.vis.clear_dataset_at", type=str, nargs='+', help='If the dreams at the beginning of the advance loop should be cleared.')
    parser.add_argument("--datamodule.vis.decorrelate", action="store_true", help='If the dreams should be decorrelated.')
    
    parser.add_argument("--model.loss.chi.sigma", type=float, default=0.1, help='How close points from latent space inside \
current batch should be close to each other. Should be lesser than model.loss.chi.rho. The smaller the less scattered points of the same class. \
It is important to know that sigma with rho directly affect convergence. The bigger the difference between sigma and rho the lesser must be the learing rate \
at least in case of SGD or there will be exploding gradient.')
    #    parser.add_argument("--model.loss.chi.rho", type=float, default=1., help='How far means from different targets \
    #should be appart from each other. Should be greather than model.loss.chi.sigma. The larger it is, the more scattered the points of different classes.')
    
    parser.add_argument("--model.loss.chi.ratio", type=float, default=2.5, help="The ratio of rho/sigma in loss function. It should be bigger than \
1. to be able to learn. The greater the absolute value, the clearer the divisions between points from the same class and points \
from different classes become. If the value is greater than 1, points from the same classes start to approach each other and \
points from different classes start to move away. For a value less than 1 the reverse relationship occurs.")
    parser.add_argument("--model.loss.chi.scale", type=float, default=2, help="The greater the scale the flatter curve of loss function. \
Increasing its value can help to eliminate exploding gradient problem while having high enough learning rate.")
    parser.add_argument("--model.loss.chi.l2", type=float, default=1e-3, help="L2 regularization for ChiLossV2 weights of points means normalization.")
    parser.add_argument("--model.loss.chi.shift_min_distance", type=float, default=10., help="")
    parser.add_argument("--model.loss.chi.shift_std_of_mean", type=float, default=15., help="")
    parser.add_argument("--model.loss.chi.ratio_gamma", type=float, default=1., help="")
    parser.add_argument("--model.loss.chi.scale_gamma", type=float, default=1., help="")
    parser.add_argument("--model.loss.chi.ratio_milestones", nargs='+', type=float, help="")
    parser.add_argument("--model.loss.chi.scale_milestones", nargs='+', type=float, help="")

    parser.add_argument("--model.loss.chi.dual.inner_scale", type=float, default=1., help="For use chi loss.")
    parser.add_argument("--model.loss.chi.dual.outer_scale", type=float, default=1., help="For use only cross entropy.")

    parser.add_argument("--loop.vis.layerloss.deep_inversion.use_at", type=str, nargs='+', help='Regularization variance of the input dream image.')
    parser.add_argument("--loop.vis.layerloss.deep_inversion.scale", type=float, default=1., help='')
    parser.add_argument("--loop.vis.layerloss.deep_inversion.scale_file", type=str, help='')
    parser.add_argument("--loop.vis.layerloss.deep_inversion.hook_to", nargs='+', type=str, help='')

    parser.add_argument("--loop.vis.layerloss.deep_inversion_target.use_at", type=str, nargs='+', help='')
    parser.add_argument("--loop.vis.layerloss.deep_inversion_target.scale", type=float, default=1., help='')
    parser.add_argument("--loop.vis.layerloss.deep_inversion_target.hook_to", nargs='+', type=str, help='')

    parser.add_argument("--loop.vis.image_reg.var.use_at", type=str, nargs='+', help='')
    parser.add_argument("--loop.vis.image_reg.var.scale", type=float, default=2.5e-5, help='')

    parser.add_argument("--loop.vis.image_reg.l2.use_at", type=str, nargs='+', help='')   
    parser.add_argument("--loop.vis.image_reg.l2.coeff", type=float, default=1e-05, help='')

    return parser

def dual_optim_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument("--model.outer.optim.type", type=str, help='')
    parser.add_argument("--model.outer.optim.reset_type", type=str, default='default', help='')
    parser.add_argument("--model.outer.optim.kwargs.lr", type=float, help='')
    parser.add_argument("--model.outer.optim.kwargs.gamma", type=float, help='')
    parser.add_argument("--model.outer.optim.kwargs.momentum", type=float, help='')
    parser.add_argument("--model.outer.optim.kwargs.dampening", type=float, help='')
    parser.add_argument("--model.outer.optim.kwargs.weight_decay", type=float, help='')
    parser.add_argument("--model.outer.optim.kwargs.betas", nargs='+', type=float, help='')
    parser.add_argument("--model.outer.optim.kwargs.amsgrad", help='')

    parser.add_argument("--model.outer.sched.type", type=str, help='')
    parser.add_argument("--model.outer.sched.kwargs.gamma", type=float)
    parser.add_argument("--model.outer.sched.kwargs.milestones", nargs='+', type=int)
    parser.add_argument("--model.outer.sched.steps", nargs='+', type=int, help='')

    parser.add_argument("--model.inner.cfg.partial_backward", action="store_true", help='')
    parser.add_argument("--model.inner.cfg.visualize_type",type=str, default='outer', help='')
    parser.add_argument("--model.inner.first.optim.type", type=str, help='')
    parser.add_argument("--model.inner.first.optim.reset_type", type=str, default='default', help='')
    parser.add_argument("--model.inner.first.optim.kwargs.lr", type=float, help='')
    parser.add_argument("--model.inner.first.optim.kwargs.gamma", type=float, help='')
    parser.add_argument("--model.inner.first.optim.kwargs.momentum", type=float, help='')
    parser.add_argument("--model.inner.first.optim.kwargs.dampening", type=float, help='')
    parser.add_argument("--model.inner.first.optim.kwargs.weight_decay", type=float, help='')
    parser.add_argument("--model.inner.first.optim.kwargs.betas", nargs='+', type=float, help='')
    parser.add_argument("--model.inner.first.optim.kwargs.amsgrad", help='')

    parser.add_argument("--model.inner.first.sched.type", type=str, help='')
    parser.add_argument("--model.inner.first.sched.kwargs.gamma", type=float)
    parser.add_argument("--model.inner.first.sched.kwargs.milestones", nargs='+', type=int)
    parser.add_argument("--model.inner.first.sched.steps", nargs='+', type=int, help='')

    parser.add_argument("--model.inner.second.optim.type", type=str, help='')
    parser.add_argument("--model.inner.second.optim.reset_type", type=str, default='default', help='')
    parser.add_argument("--model.inner.second.optim.kwargs.lr", type=float, help='')
    parser.add_argument("--model.inner.second.optim.kwargs.gamma", type=float, help='')
    parser.add_argument("--model.inner.second.optim.kwargs.momentum", type=float, help='')
    parser.add_argument("--model.inner.second.optim.kwargs.dampening", type=float, help='')
    parser.add_argument("--model.inner.second.optim.kwargs.weight_decay", type=float, help='')
    parser.add_argument("--model.inner.second.optim.kwargs.betas", nargs='+', type=float, help='')
    parser.add_argument("--model.inner.second.optim.kwargs.amsgrad", help='')

    parser.add_argument("--model.inner.second.sched.type", type=str, help='')
    parser.add_argument("--model.inner.second.sched.kwargs.gamma", type=float)
    parser.add_argument("--model.inner.second.sched.kwargs.milestones", nargs='+', type=int)
    parser.add_argument("--model.inner.second.sched.steps", nargs='+', type=int, help='')

    return parser

def model_statistics_optim_arg_parser(parser: ArgumentParser) -> ArgumentParser:
    ######################################
    ######          other           ######
    ######################################
    parser.add_argument("--pca_estimate_rank", type=int, default=6, 
        help='Slighty overestimated rank of input matrix in PCA algorithm. Default is 6.')
    parser.add_argument("--stat.compare_latent", action="store_true", help='')
    parser.add_argument("--stat.disorder_dream", action="store_true", help='')
    parser.add_argument("--stat.limit_plots_to", type=int, default=6, help='')
    parser.add_argument("--stat.collect_stats.enable", action="store_true", help='')
    parser.add_argument("--stat.collect_stats.use_dream_dataset", action="store_true", help='')
    parser.add_argument("--stat.plot_classes", nargs='+', type=str, help='Classes to plot in 2d plot. \
For multi dims they will be plotted in paris for all combinations of dims.')
    parser.add_argument("--stat.disorder.sigma", type=float, default=0.0, help='')
    parser.add_argument("--stat.disorder.start_img_val", type=float, help='Default None')
    parser.add_argument("--loop.save.ignore_config", action="store_true", help='')

    parser.add_argument("--stat.collect.latent_buffer.enable", action="store_true", help='')
    parser.add_argument("--stat.collect.latent_buffer.name", type=str, help='Default None')
    parser.add_argument("--stat.collect.latent_buffer.cl_idx", type=int, help='')
    parser.add_argument("--stat.collect.latent_buffer.size", type=int, default=50, help='')

    parser.add_argument("--stat.collect.single_dream.enable", action="store_true", help='')
    parser.add_argument("--stat.collect.single_dream.sigma", type=float, default=0.0, help='')

    return parser

def parse_args(parser: ArgumentParser, args: Sequence[str] | None = None, namespace: None = None) -> Namespace:
    args = parser.parse_args(args, namespace)

    args = load_config(args, parser)
    export_config(args)

    args = convert_args_str_to_list_int(args)

    args = extend_namespace(args)
    
    return args

def extend_namespace(args):
    """
        Split arguments keys by '.' character. Create an hierarchy of Namespace objects with specified names.
    """
    namespace = vars(args)
    for k in list(namespace.keys()):
        if('.' in k):
            data = k.split('.')
            current = args
            for d in data[:-1]:
                if(hasattr(current, d)):
                    if(isinstance(getattr(current, d), Namespace)):
                        current = getattr(current, d)
                    else:
                        raise Exception(f'Bad name: {data} == {namespace[k]}. Search for {d}= in Namespace.')
                else:
                    setattr(current, d, Namespace())
                    current = getattr(current, d)

            setattr(current, data[-1], namespace[k])
            delattr(args, k) 

    return args

def convert_args_str_to_list_int(args: Namespace):
    to_check = [
        'loop.train_at', 
        'loop.vis.generate_at', 
        'datamodule.vis.only_vis_at', 
        'loop.model.reload_at', 
        'loop.model.reinit_at',
    ]
    for k in args.__dict__.keys():
        if(str.endswith(k, '_at') and k not in to_check):
            to_check.append(k)
    for k in to_check:
        v = args.__dict__[k]
        if(v is None):
            continue
        if(isinstance(v, str)):
            b = utils.str_is_true(v)
            if(b is not None):
                v = b
        elif(isinstance(v, list)):
            new_list = []
            for vv in v:
                b = utils.str_to_bool_int(vv)
                if(b is not None):
                    new_list.append(b)
            v = new_list
        args.__dict__[k] = v
    return args

def attack_args_to_kwargs(args):
    if(args.model.robust.enable):
        return {
            "constraint": args.model.robust.kwargs.constraint,
            "eps": args.model.robust.kwargs.eps,
            "step_size": args.model.robust.kwargs.step_size,
            "iterations": args.model.robust.kwargs.iterations,
            "random_start": args.model.robust.kwargs.random_start,
            "custom_loss": args.model.robust.kwargs.custom_loss,
            "random_restarts": args.model.robust.kwargs.random_restart,
            "use_best": args.model.robust.kwargs.use_worst,
            'with_latent': args.model.robust.kwargs.with_latent,
            'fake_relu': args.model.robust.kwargs.fake_relu,
            'no_relu':args.model.robust.kwargs.no_relu,
            'make_adv':args.model.robust.kwargs.make_adversary,
        }
    else:
        return {}

def log_to_wandb(args):
    #if(wandb.run is not None): # does not work
    wandb.config.update({'Plain args': str(sys.argv)})
    wandb.config.update(args, allow_val_change=True)
    pp.sprint(f'{pp.COLOR.NORMAL}\tInput command line:')
    print(' '.join(sys.argv))
    pp.sprint(f'{pp.COLOR.NORMAL}\tUsed config:')
    print(wandb.config)

def wandb_run_name(args, id):
    dream = "dull_"
    tr = ""
    if(args.datamodule.vis.only_vis_at is not None and args.datamodule.vis.only_vis_at != False):
        dream = 'dream_'
        if(args.datamodule.vis.disable_transforms != True):
            tr = "tr_"
    text = f"{args.config.framework_type}_{dream}{tr}"
    if(args.loop.vis.layerloss.mean_norm.use_at is not None and args.datamodule.vis.only_vis_at != False):
        text = f"{text}ll_mean_norm{args.loop.vis.layerloss.mean_norm.scale}_"
    if(args.model.layer_replace.enable):
        text = f"{text}layer_replace_"
    if(args.loop.vis.layerloss.grad_pruning.use_at is not None and args.loop.vis.layerloss.grad_pruning.use_at != False):
        text = f"{text}gp{args.loop.vis.layerloss.grad_pruning.percent}_"
    if(args.loop.vis.layerloss.grad_activ_pruning.use_at is not None and args.loop.vis.layerloss.grad_activ_pruning.use_at != False):
        text = f"{text}gap{args.loop.vis.layerloss.grad_activ_pruning.percent}_"
    
    return f"{id}_{text}"

def load_config(args: Namespace, parser: ArgumentParser, filepath:str=None) -> Namespace:
    """
        Load config file as defaults arguments. Arguments from command line have priority. 
    """
    if args.__dict__['config.load'] is not None:
        for conf_fname in args.__dict__['config.load']:
            if(filepath is None):
                folder = Path(args.__dict__['config.folder'])
            else:
                folder = filepath if not isinstance(filepath, Path) else Path(filepath)
            file = folder / conf_fname
            file = glob.glob(str(file), recursive=True)
            if(len(file) != 1):
                raise Exception(f'Cannot load config - no or too many matching filenames. From "{conf_fname}" found only these paths: {file}')
            file = file[0]
            with open(file, 'r') as f:
                parser.set_defaults(**json.load(f))
        
        args = parser.parse_args()
    return args

def can_export_config(args) -> bool:
    fast_dev_run = 'fast_dev_run.enable' in args.__dict__
    if(not fast_dev_run):
        if(not hasattr(args, 'fast_dev_run')):
            raise Exception("No argument args.fast_dev_run")
        fast_dev_run = args.fast_dev_run.enable
    else:
        fast_dev_run = args.__dict__.get('fast_dev_run.enable')

    config_export = 'config.export' in args.__dict__
    if(not config_export):
        if(not hasattr(args, 'config')):
            raise Exception("No argument args.config")
        config_export = args.config.export
    else:
        config_export = args.__dict__.get('config.export')
        
    return config_export and not fast_dev_run

def export_config(args: Namespace, filepath:str=None, filename:str=None) -> None:
    if can_export_config(args) or filename is not None:
        tmp_args = vars(args).copy()
        if(args.__dict__.get('config.export')):
            filename = args.__dict__.get('config.export') if filename is None else filename
            del tmp_args['config.export']  # Do not dump value of conf_export flag
            del tmp_args['config.load']
            del tmp_args['fast_dev_run.enable']  # Fast dev run should not be present in config file

            # Remove all options that are corelated to saving and loading
            del tmp_args['loop.load.dreams']  
            del tmp_args['loop.load.model'] 
            del tmp_args['loop.load.layer_stats'] 

            del tmp_args['loop.save.dreams']  
            del tmp_args['loop.save.model'] 
            del tmp_args['loop.save.layer_stats'] 
            del tmp_args['loop.save.enable_checkpoint'] 
            if(filepath is None):
                path = path = Path(args.__dict__['config.folder']) / filename
            else:
                path = filepath if not isinstance(filepath, Path) else Path(filepath) / filename
        else:
            filename = args.config.export if filename is None else filename
            if(filepath is None):
                path = path = Path(args.config.folder) / filename
            else:
                path = filepath if not isinstance(filepath, Path) else Path(filepath) / filename

        Path.mkdir(path.parent, parents=True, exist_ok=True)
        dump = json.dumps(tmp_args, indent=4, sort_keys=True, cls=NamespaceEncoder)
        with open(path, 'w') as f:
            f.write(dump)

def get_arg(args: Namespace, path: str) -> None|Any:
    """
    Get argument by value from extended Namespace.
    """
    #args.model.robust.enable
    for s in path.split('.'):
        try:
            args = getattr(args, s)
        except Exception:
            return None
    return args

class NamespaceEncoder(JSONEncoder):
    def default(self, obj):
        if(isinstance(obj, Namespace)):
            return obj.__dict__
        return JSONEncoder.default(self, obj)
