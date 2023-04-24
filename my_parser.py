from argparse import ArgumentParser, Namespace
import wandb
import sys
import json
from pathlib import Path
import glob
import os
from utils import utils
import numpy as np
import functools as ft

def arg_parser() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(prog='Continual dreaming', add_help=True, description='Configurable framework to work with\
continual learning tasks.', 
epilog="""
Validation dataset uses data from test dataset but divided into appropriate tasks.
Test dataset uses test data with only the classes used in previous tasks. 
""")

    parser.add_argument("--config.seed", type=int, help='Seed of the pytorch random number generator. Default None - random seed.') ##**
    parser.add_argument("--loop.run_training_at", nargs='+', type=str, default='True', help='Run training at corresponding loop index or indexes. \
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
    parser.add_argument("--loss.chi.buffer_size_per_class", type=int, default=40, help='Size per class of the cyclic \
buffer used in island overlay for model.')

    parser.add_argument("--datamodule.disable_shuffle", action="store_true", help='Flag to shuffle train normal and dream datasets. If \
flag "dataloader_disable_dream_shuffle" is set then it takes precedence over this flag.')
    # worker numbers
    parser.add_argument("--datamodule.num_workers", type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument("--datamodule.vis.num_workers", type=int, help='Number of visualization dataloader workers.')
    parser.add_argument("--datamodule.test_num_workers", type=int, help='Number of test dataloader workers.')
    parser.add_argument("--datamodule.val_num_workers", type=int, help='Number of validation dataloader workers.')

    parser.add_argument("--model.optim.type", type=str, default='adam', help='')
    parser.add_argument("--model.scheduler.type", type=str, default='none', help='Type of scheduler. Use "model.scheduler_steps" \
to choose epoch at which to call it.')
    parser.add_argument("--model.optim.reset_type", type=str, default='default', help='')

    parser.add_argument("--loop.save.model", action="store_true", help='Save model at the end of all training loops') ##**
    parser.add_argument("--loop.save.model_inner_path", type=str, help='') ##**
    parser.add_argument("--loop.load.model", type=str, help='') ##**
    parser.add_argument("--loop.save.enable_checkpoint", action="store_true", help='')
    parser.add_argument("--loop.save.export_path", type=str, help='')
    parser.add_argument("--loop.load.export_path", type=str, help='')
    parser.add_argument("--loop.save.dreams", type=str, help='') ##**
    parser.add_argument("--loop.load.dreams", type=str, help='') ##**
    
    ######################################
    #####    numerical parameters   ######
    ######################################
    parser.add_argument("--datamodule.batch_size", type=int, default=32, help='Size of the batch.')
    parser.add_argument("--config.num_tasks", type=int, default=1, help='How many tasks will be created')
    parser.add_argument("--loop.quantity", type=int, default=1, help='How many loops will be traversed. \
Each new loop will increment current task index. If "num_loops">"num_tasks" then at each new loop \
the last task in array will be used.') ##**
    parser.add_argument("--config.epochs_per_task", nargs='+', type=int, help='How many epochs do per one task in "num_tasks". \
Array size should be the same as num_loops for each loop.') ##**
    parser.add_argument("--config.number_of_classes", type=int, default=10, help='Number of classes model should output. \
If less than in dataset then model will be trained and validated only using this number of classes')

    parser.add_argument("--optim.lr", type=float, default=1e-3, help='Learning rate of the optimizer.')
    parser.add_argument("--optim.gamma", type=float, default=1, help='Gamma parameter for optimizer if exist.')

    parser.add_argument("--model.norm_lambda", type=float, default=0., help='Lambda parametr of the used l2 normalization. If 0. then \
no normalization is used. Normalization is used to the last model layer, the latent output of the "CLModelWithIslands".')
    parser.add_argument("--model.scheduler_steps", nargs='+', type=int, default=(3, ), help='Epoch training steps \
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
    parser.add_argument("--config.model_type", type=str, help='Model type') ##**

    ######################################
    #####     dream parameters      ######
    ######################################
    parser.add_argument("--datamodule.vis.generate_at", nargs='+', type=str, default='False', help='At which loop or loops enable dreaming where framework \
should produce dreams and use them during training. Can take one or more indexes and boolean. Default None will not produce any dreams. \
Without running this and running "loop.run_training_at" will run only test. Use this with "datamodule.vis.only_vis_at" to train only on dream batch.') ##**
    parser.add_argument("--datamodule.vis.per_target", type=int, default=128, help='How many epochs do per one task in "num_tasks"') ##**
    parser.add_argument("--datamodule.vis.batch_size", type=int, default=128, help='How many images \
in batch during dreaming should be produced.')
    parser.add_argument("--datamodule.vis.optim.lr", type=float, default=1e-3, help='Learning rate of the dream optimizer.')
    parser.add_argument("--datamodule.vis.threshold", nargs='+', type=int, default=[1024, ], help='How many iterations should \
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
    
    parser.add_argument("--loss.chi.sigma", type=float, default=0.01, help='How close points from latent space inside \
current batch should be close to each other. Should be lesser than loss.chi.rho. The smaller the less scattered points of the same class.')
    parser.add_argument("--loss.chi.rho", type=float, default=1., help='How far means from different targets \
should be appart from each other. Should be greather than loss.chi.sigma. The larger it is, the more scattered the points of different classes.')

    parser.add_argument("--loop.vis.use_input_img_var_reg_at", type=str, nargs='+', help='Regularization variance of the input dream image.')
    parser.add_argument("--loop.vis.use_var_img_reg_at", type=str, nargs='+', help='')
    parser.add_argument("--loop.vis.use_l2_img_reg_at", type=str, nargs='+', help='')
    parser.add_argument("--loop.vis.bn_reg_scale", type=float, default=1e2, help='')
    parser.add_argument("--loop.vis.var_scale", type=float, default=2.5e-5, help='')
    parser.add_argument("--loop.vis.l2_coeff", type=float, default=1e-05, help='')


    ######################################
    #####       model reinit        ######
    ######################################
    parser.add_argument("--loop.reload_model_at", nargs='+', type=str, help='Reload model weights after each main loop.\
Reloading means any weights that model had before training will be reloaded. Reload is done AFTER dream generation if turned on.') ##**
    parser.add_argument("--loop.reinit_model_at", nargs='+', type=str, help='Reset model after each main loop.\
Model will have newly initialized weights after each main loop. Reinit is done AFTER dream generation if turned on.') ##**

    parser.add_argument("--loop.weight_reset_sanity_check", action="store_true", help='Enable sanity check for reload/reinit weights.')



    ######################################
    #####       fast dev run        ######
    ######################################
    parser.add_argument("-f", "--fast-dev-run", action="store_true", help='Use to fast check for errors in code.\
It ') ##**
    parser.add_argument("--fast_dev_run_batches", type=int, default=30, help='')
    parser.add_argument("--fast_dev_run_epochs", type=int, default=1, help='')
    parser.add_argument("--fast_dev_run_dream_threshold", nargs='+', type=int, default=[32], help='')


    ######################################
    ######         robust           ######
    ######################################
    parser.add_argument("--enable_robust", action="store_true", help='Enable robust training.')
    parser.add_argument("--attack_constraint", type=str, default="2", help='')
    parser.add_argument("--attack_eps", type=float, default=0.5, help='')
    parser.add_argument("--attack_step_size", type=float, default=1.5, help='')
    parser.add_argument("--attack_iterations", type=int, default=10, help='')
    parser.add_argument("--attack_random_start", type=int, default=0, help='')
    parser.add_argument("--attack_random_restarts", type=int, default=0, help='')
    parser.add_argument("--attack_custom_loss", type=str, help='') # default None
    parser.add_argument("--attack_use_worst", action="store_false", help='')
    parser.add_argument("--attack_with_latent", action="store_true", help='')
    parser.add_argument("--attack_fake_relu", action="store_true", help='')
    parser.add_argument("--attack_no_relu", action="store_true", help='')
    parser.add_argument("--attack_make_adv", action="store_true", help='') # required

    ######################################
    #####     layer statistics      ######
    ######################################
    parser.add_argument("--loop.gather_layer_loss_at", type=int, help='Gather layer statistics \
at given fit loop index. Default None means this functionality is not enabled. Value less than zero \
means it will be used just like the python indexing for negative numbers.') ##**
    parser.add_argument("--loop.use_layer_loss_at", type=int, help='Use layer loss function \
at given fit loop index. Default None means this functionality is not enabled. Value less than zero \
means it will be used just like the python indexing for negative numbers.') ##**
    parser.add_argument("--loop.save_layer_stats", type=str, help='Path where to save layer_stats.') ##**
    parser.add_argument("--loop.load_layer_stats", type=str, help='Path from where to load layer_stats.') ##**
    parser.add_argument("--loop.layer_stats_hook_to", nargs='+', type=str, help='Name of the layers to hook up. If exception \
thrown, list of avaliable layers will be displayed.') ##**
    parser.add_argument("--model.replace_layer", action='store_true', help='Replace layer. For now replace ReLu to GaussA.') ##**
    parser.add_argument("--loop.layerloss.scaling", type=float, default=0.001, help='Scaling for layer loss.') ##**
    parser.add_argument("--loop.use_grad_pruning_at", type=int, help='Use gradient pruning at.') ##**
    parser.add_argument("--loop.grad_pruning_percent", type=float, default=0.01,
        help='Percent of gradient pruning neurons at given layer. Selected by std descending.') ##**
    parser.add_argument("--loop.use_grad_activ_pruning_at", type=int, help='') ##**
    parser.add_argument("--loop.grad_activ_pruning_percent", type=float, default=0.01, help='') ##**
    parser.add_argument("--loop.layerloss.del_cov_after", action="store_true", help='Delete covariance matrix after calculating inverse of covariance.') ##**

    ######################################
    ######          other           ######
    ######################################
    parser.add_argument("--config.save_folder", type=str, default='run_conf', 
        help='Folder where to save and load argparse config for flags "config" and "config.export" ')
    parser.add_argument("--pca_estimate_rank", type=int, default=6, 
        help='Slighty overestimated rank of input matrix in PCA algorithm. Default is 6.')
    parser.add_argument("--stat.compare_latent", action="store_true", help='')
    parser.add_argument("--stat.disorder_dream", action="store_true", help='')
    parser.add_argument("--stat.collect_stats", action="store_true", help='')


    parser.add_argument("--test1.test2.test3", action="store_true", help='')

    args = parser.parse_args()

    args = load_config(args, parser)
    export_config(args)

    args = convert_args_str_to_list_int(args)

    args = extend_namespace(args)
    
    return args, parser

def extend_namespace(args):
    namespace = vars(args)
    new_dict = {}
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


            ###
            #data = k.split('.')
            #current_dict = new_dict
            #for d in data[:-1]:
            #    if(d not in current_dict):
            #        current_dict[k] = {}
            #    else:
            #        current_dict = current_dict[k]
            #
            #if(data[-1] not in current_dict):
            #    current_dict[k] = namespace[k]
            #else:
            #    raise Exception('Data already in dictionary')

            ###
            #data = k.split('.')
            #current = namespace[k]
            #for d in reversed(data[1:]):
            #    current = Namespace(**{d: current})
            #setattr(args, data[0], current)
            delattr(args, k) 

    #print()
    #print(args)
    #exit()
    return args

def convert_args_str_to_list_int(args: Namespace):
    to_check = [
        'loop.run_training_at', 
        'datamodule.vis.generate_at', 
        'datamodule.vis.only_vis_at', 
        'loop.reload_model_at', 
        'loop.reinit_model_at',
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
    if(args.enable_robust):
        return {
            "constraint": args.attack_constraint,
            "eps": args.attack_eps,
            "step_size": args.attack_step_size,
            "iterations": args.attack_iterations,
            "random_start": args.attack_random_start,
            "custom_loss": args.attack_custom_loss,
            "random_restarts": args.attack_random_restarts,
            "use_best": args.attack_use_worst,
            'with_latent': args.attack_with_latent,
            'fake_relu': args.attack_fake_relu,
            'no_relu':args.attack_no_relu,
            'make_adv':args.attack_make_adv,
        }
    else:
        return {}

def optim_params_to_kwargs(args):
    return {
        'lr': args.optim.lr,
        'gamma':args.optim.gamma,
    }

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
    text = f"{args.config.model_type}_{dream}{tr}"
    if(args.loop.use_layer_loss_at is not None and args.datamodule.vis.only_vis_at != False):
        text = f"{text}ll{args.loop.layerloss.scaling}_"
    if(args.model.replace_layer):
        text = f"{text}replayer_"
    if(args.loop.use_grad_pruning_at is not None and args.loop.use_grad_pruning_at != False):
        text = f"{text}gp{args.loop.grad_pruning_percent}_"
    if(args.loop.use_grad_activ_pruning_at is not None and args.loop.use_grad_activ_pruning_at != False):
        text = f"{text}gap{args.loop.grad_activ_pruning_percent}_"
    
    return f"{text}{np.random.randint(0, 5000)}"

def load_config(args: Namespace, parser: ArgumentParser) -> Namespace:
    """
        Load config file as defaults arguments. Arguments from command line have priority. 
    """
    if args.__dict__['config.load'] is not None:
        for conf_fname in args.__dict__['config.load']:
            folder = Path(args.__dict__['config.save_folder'])
            file = folder / conf_fname
            file = glob.glob(str(file), recursive=True)
            if(len(file) != 1):
                raise Exception(f'Cannot load config - no or too many matching filenames. From "{conf_fname}" found only these paths: {file}')
            file = file[0]
            with open(file, 'r') as f:
                parser.set_defaults(**json.load(f))
        
        args = parser.parse_args()
    return args

def export_config(args: Namespace) -> None:
    if args.__dict__['config.export'] and not args.fast_dev_run:
        tmp_args = vars(args).copy()
        del tmp_args['config.export']  # Do not dump value of conf_export flag
        del tmp_args['config']  # Values already loaded
        del tmp_args['fast_dev_run']  # Fast dev run should not be present in config file

        # Remove all options that are corelated to saving and loading
        del tmp_args['loop.load.dreams']  
        del tmp_args['loop.save.dreams']  
        del tmp_args['loop.export_path'] 
        del tmp_args['loop.save.model_inner_path'] 
        del tmp_args['loop.load.model'] 
        del tmp_args['loop.save.model'] 
        del tmp_args['loop.save.enable_checkpoint'] 
        path = Path(args.__dict__['config.save_folder']) / args.config.export
        Path.mkdir(path.parent, parents=True, exist_ok=True)
        dump = json.dumps(tmp_args,  indent=4, sort_keys=True)
        with open(path, 'w') as f:
            f.write(dump)