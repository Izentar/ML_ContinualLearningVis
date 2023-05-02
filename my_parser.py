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

def arg_parser() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(prog='Continual dreaming', add_help=True, description='Configurable framework to work with\
continual learning tasks.', 
epilog="""
Validation dataset uses data from test dataset but divided into appropriate tasks.
Test dataset uses test data with only the classes used in previous tasks. 
""")


    parser.add_argument("--wandb.run.folder", type=str, default="log_run/", help='') ##**
    parser.add_argument("--config.seed", type=int, help='Seed of the pytorch random number generator. Default None - random seed.') ##**
    parser.add_argument("--config.folder", type=str, default='run_conf/', help='Root forlder of the runs.') ##**
    parser.add_argument("--loop.train_at", nargs='+', type=str, default='True', help='Run training at corresponding loop index or indexes. \
If True, run on all loops, if False, do not run. If "loop.gather_layer_loss_at" is set, then it takes precedence.') ##**
    parser.add_argument("--config.load", action="append", help='The config file(s) that should be used. The config file \
takes precedence over command line arguments. Config files will be applied in order of the declaration.')
    parser.add_argument("--config.export", type=str, help='File where to export current config.')
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

    parser.add_argument("--model.optim.type", type=str, default='adam', help='')
    parser.add_argument("--model.sched.type", type=str, default='none', help='Type of scheduler. Use "model.scheduler.steps" \
to choose epoch at which to call it.')
    parser.add_argument("--model.optim.reset_type", type=str, default='default', help='')

    parser.add_argument("--loop.save.model", action="store_true", help='Save model at the end of all training loops') ##**
    parser.add_argument("--loop.load.model", action="store_true", help='') ##**
    parser.add_argument("--loop.save.enable_checkpoint", action="store_true", help='')
    parser.add_argument("--loop.save.dreams", action="store_true", help='') ##**
    parser.add_argument("--loop.load.dreams", action="store_true", help='') ##**
    parser.add_argument("--loop.save.root", type=str, help='') ##**
    parser.add_argument("--loop.load.root", type=str, help='') ##**
    parser.add_argument("--loop.load.id", type=str, help='') ##**
    
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

    parser.add_argument("--model.optim.kwargs.lr", type=float, default=1e-3, help='Learning rate of the optimizer.')
    parser.add_argument("--model.optim.kwargs.gamma", type=float, default=1, help='Gamma parameter for optimizer if exist.')

    parser.add_argument("--model.norm_lambda", type=float, default=0., help='Lambda parametr of the used l2 normalization. If 0. then \
no normalization is used. Normalization is used to the last model layer, the latent output of the "CLModelWithIslands".')
    parser.add_argument("--model.sched.steps", nargs='+', type=int, default=(3, ), help='Epoch training steps \
at where to call scheduler, change learning rate. Use "model.scheduler.type" to enable scheduler.')
    

    ######################################
    #####       configuration       ######
    ######################################
    parser.add_argument("--config.framework_type", type=str, default='sae-chiloss', help='Framework type') ##**
    parser.add_argument("--config.dream_obj_type", nargs='+', type=str, help='Model objective funtions type. May be multiple')
    parser.add_argument("--config.select_task_type", type=str, help='From utils.functional.select_task.py')
    parser.add_argument("--config.target_processing_type", type=str, help='From utils.functional.target_processing.py')
    parser.add_argument("--config.task_split_type", type=str, help='From utils.functional.task_split.py')
    parser.add_argument("--config.overlay_type", type=str, help='Overlay type')
    parser.add_argument("--model.type", type=str, help='Model type') ##**

    ######################################
    #####     dream parameters      ######
    ######################################
    parser.add_argument("--loop.vis.generate_at", nargs='+', type=str, default='False', help='At which loop or loops enable dreaming where framework \
should produce dreams and use them during training. Can take one or more indexes and boolean. Default None will not produce any dreams. \
Without running this and running "loop.train_at" will run only test. Use this with "datamodule.vis.only_vis_at" to train only on dream batch.') ##**
    parser.add_argument("--datamodule.vis.per_target", type=int, default=128, help='How many epochs do per one task in "num_tasks"') ##**
    parser.add_argument("--datamodule.vis.batch_size", type=int, default=128, help='How many images \
in batch during dreaming should be produced.')
    parser.add_argument("--datamodule.vis.optim.kwargs.lr", type=float, default=1e-3, help='Learning rate of the dream optimizer.')
    parser.add_argument("--datamodule.vis.sched.type", type=str)
    parser.add_argument("--datamodule.vis.threshold", nargs='+', type=int, default=[512, ], help='How many iterations should \
be used to generate an output image during dreaming, using only max value. Values lesser than max are points where the \
images from the batch will be additionaly saved.')
    parser.add_argument("--datamodule.vis.disable_transforms", action="store_true", help='Enable and add all default \
tranforms on dreamed images used in lucid framework in main function.') ##**
    parser.add_argument("--datamodule.vis.disable_shuffle", action="store_true", help='Flag to shuffle only train dream dataset')
    parser.add_argument("--datamodule.vis.image_type", type=str, default='fft', help='Type of image. Default \
"fft"; "pixel"; "cppn"(does not use "datamodule.vis.batch_size")')
    parser.add_argument("--datamodule.vis.only_vis_at", nargs='+', type=str, default='False', help='Use this flag to train only on dream batch \
after first epoch when dream batch is created.') ##**
    parser.add_argument("--datamodule.vis.standard_image_size", nargs='+', type=int, help='Tuple of sizes of the image after image transformation during dreaming. \
Checks if the output image has the provided shape. Do not include batch here. Default None.') ##**
    parser.add_argument("--loop.vis.clear_dataset_at", type=str, nargs='+', help='If the dreams at the beginning of the advance loop should be cleared.')
    parser.add_argument("--datamodule.vis.decorrelate", action="store_true", help='If the dreams should be decorrelated.')
    
    parser.add_argument("--model.loss.chi.sigma", type=float, default=0.01, help='How close points from latent space inside \
current batch should be close to each other. Should be lesser than model.loss.chi.rho. The smaller the less scattered points of the same class.')
    parser.add_argument("--model.loss.chi.rho", type=float, default=1., help='How far means from different targets \
should be appart from each other. Should be greather than model.loss.chi.sigma. The larger it is, the more scattered the points of different classes.')

    parser.add_argument("--loop.vis.layerloss.deep_inversion.use_at", type=str, nargs='+', help='Regularization variance of the input dream image.')
    parser.add_argument("--loop.vis.layerloss.deep_inversion.scale", type=float, default=1e2, help='')
    parser.add_argument("--loop.vis.layerloss.deep_inversion.hook_to", nargs='+', type=str, help='')

    parser.add_argument("--loop.vis.image_reg.var.use_at", type=str, nargs='+', help='')
    parser.add_argument("--loop.vis.image_reg.var.scale", type=float, default=2.5e-5, help='')

    parser.add_argument("--loop.vis.image_reg.l2.use_at", type=str, nargs='+', help='')   
    parser.add_argument("--loop.vis.image_reg.l2.coeff", type=float, default=1e-05, help='')

    parser.add_argument("--datamodule.vis.optim.type", type=str, default='adam', help='')


    ######################################
    #####       model reinit        ######
    ######################################
    parser.add_argument("--loop.model.reload_at", nargs='+', type=str, help='Reload model weights after each main loop.\
Reloading means any weights that model had before training will be reloaded. Reload is done AFTER dream generation if turned on.') ##**
    parser.add_argument("--loop.model.reinit_at", nargs='+', type=str, help='Reset model after each main loop.\
Model will have newly initialized weights after each main loop. Reinit is done AFTER dream generation if turned on.') ##**

    parser.add_argument("--loop.weight_reset_sanity_check", action="store_true", help='Enable sanity check for reload/reinit weights.')



    ######################################
    #####       fast dev run        ######
    ######################################
    parser.add_argument("-f", "--fast_dev_run.enable", action="store_true", help='Use to fast check for errors in code.\
It ') ##**
    parser.add_argument("--fast_dev_run.batches", type=int, default=30, help='')
    parser.add_argument("--fast_dev_run.epochs", type=int, default=1, help='')
    parser.add_argument("--fast_dev_run.vis_threshold", nargs='+', type=int, default=[32], help='')


    ######################################
    ######         robust           ######
    ######################################
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

    ######################################
    #####     layer statistics      ######
    ######################################
    parser.add_argument("--loop.layer_stats.use_at", type=int, help='Gather layer statistics \
at given fit loop index. Default None means this functionality is not enabled. Value less than zero \
means it will be used just like the python indexing for negative numbers.') ##**
    parser.add_argument("--loop.vis.layerloss.mean_norm.use_at", type=int, help='Use layer loss function \
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
    parser.add_argument("--loop.vis.layerloss.mean_norm.scaling", type=float, default=0.001, help='Scaling for layer loss.') ##**
    parser.add_argument("--loop.vis.layerloss.grad_pruning.use_at", type=int, help='Use gradient pruning at.') ##**
    parser.add_argument("--loop.vis.layerloss.grad_pruning.percent", type=float, default=0.01,
        help='Percent of gradient pruning neurons at given layer. Selected by std descending.') ##**
    parser.add_argument("--loop.vis.layerloss.grad_activ_pruning.use_at", type=int, help='') ##**
    parser.add_argument("--loop.vis.layerloss.grad_activ_pruning.percent", type=float, default=0.01, help='') ##**
    parser.add_argument("--loop.vis.layerloss.mean_norm.del_cov_after", action="store_true", help='Delete covariance matrix after calculating inverse of covariance.') ##**

    parser.add_argument("--loop.vis.layerloss.mean_norm.device", type=str, default='cuda', help='')
    parser.add_argument("--loop.vis.layerloss.grad_pruning.device", type=str, default='cuda', help='')
    parser.add_argument("--loop.vis.layerloss.grad_activ_pruning.device", type=str, default='cuda', help='')
    parser.add_argument("--loop.layer_stats.device", type=str, default='cuda', help='')





    ######################################
    ######          other           ######
    ######################################
    parser.add_argument("--pca_estimate_rank", type=int, default=6, 
        help='Slighty overestimated rank of input matrix in PCA algorithm. Default is 6.')
    parser.add_argument("--stat.compare_latent", action="store_true", help='')
    parser.add_argument("--stat.disorder_dream", action="store_true", help='')
    parser.add_argument("--stat.collect_stats", action="store_true", help='')

    parser.add_argument("--wandb.watch.enable", action="store_true", help='')
    parser.add_argument("--wandb.watch.log_freq", type=int, default=1000, help='')

    args = parser.parse_args()

    args = load_config(args, parser)
    export_config(args)

    args = convert_args_str_to_list_int(args)

    args = extend_namespace(args)
    
    return args, parser


def extend_namespace(args):
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
    wandb.config.update(args)
    print('Input command line:')
    print(' '.join(sys.argv))
    print('Used config:')
    print(wandb.config)

def wandb_run_name(args):
    dream = "dull_"
    tr = ""
    if(args.datamodule.vis.only_vis_at is not None and args.datamodule.vis.only_vis_at != False):
        dream = 'dream_'
        if(args.datamodule.vis.disable_transforms != True):
            tr = "tr_"
    text = f"{args.config.framework_type}_{dream}{tr}"
    if(args.loop.vis.layerloss.mean_norm.use_at is not None and args.datamodule.vis.only_vis_at != False):
        text = f"{text}ll_mean_norm{args.loop.vis.layerloss.scaling}_"
    if(args.model.layer_replace.enable):
        text = f"{text}replayer_"
    if(args.loop.vis.layerloss.grad_pruning.use_at is not None and args.loop.vis.layerloss.grad_pruning.use_at != False):
        text = f"{text}gp{args.loop.vis.layerloss.grad_pruning.percent}_"
    if(args.loop.vis.layerloss.grad_activ_pruning.use_at is not None and args.loop.vis.layerloss.grad_activ_pruning.use_at != False):
        text = f"{text}gap{args.loop.vis.layerloss.grad_activ_pruning.percent}_"
    
    return f"{text}{np.random.randint(0, 5000)}"

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
    fast_dev_run = args.__dict__.get('fast_dev_run.enable')
    if(fast_dev_run is None):
        if(not hasattr(args, 'fast_dev_run')):
            raise Exception("No argument args.fast_dev_run")
        fast_dev_run = args.fast_dev_run.enable

    config_export = args.__dict__.get('config.export')
    if(config_export is None):
        if(not hasattr(args, 'config')):
            raise Exception("No argument args.config")
        config_export = args.config.export
        
    return config_export and not fast_dev_run

def export_config(args: Namespace, filepath:str=None) -> None:
    if can_export_config(args):
        tmp_args = vars(args).copy()
        if(args.__dict__.get('config.export')):
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
                path = path = Path(args.__dict__['config.folder']) / args.__dict__['config.export']
            else:
                path = filepath if not isinstance(filepath, Path) else Path(filepath) / args.__dict__['config.export']
        else:
            if(filepath is None):
                path = path = Path(args.config.folder) / args.config.export
            else:
                path = filepath if not isinstance(filepath, Path) else Path(filepath) / args.config.export

        Path.mkdir(path.parent, parents=True, exist_ok=True)
        dump = json.dumps(tmp_args, indent=4, sort_keys=True, cls=NamespaceEncoder)
        with open(path, 'w') as f:
            f.write(dump)

class NamespaceEncoder(JSONEncoder):
    def default(self, obj):
        if(isinstance(obj, Namespace)):
            return obj.__dict__
        return JSONEncoder.default(self, obj)
