from argparse import ArgumentParser, Namespace
import wandb
import sys
import json
from pathlib import Path
import glob
import os

def arg_parser() -> tuple[Namespace, ArgumentParser]:
    parser = ArgumentParser(prog='Continual dreaming', add_help=True, description='Configurable framework to work with\
continual learning tasks.', 
epilog="""
Validation dataset uses data from test dataset but divided into appropriate tasks.
Test dataset uses test data with only the classes used in previous tasks. 
""")
    
    parser.add_argument("--config", action="append", help='The config file(s) that should be used. The config file \
takes precedence over command line arguments. Config files will be applied in order of the declaration.')
    parser.add_argument("--config_export", type=str, help='File where to export current config.')
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--model_type", type=str, help='Model type') ##**
    parser.add_argument("--framework_type", type=str, default='cl-sae-crossentropy', help='Framework type') ##**
    parser.add_argument("-d", "--dataset", type=str) ##**
    parser.add_argument("--early_finish_at", type=int, default=-1, help='Finish training loop at desired epoch. Default "-1"')
    parser.add_argument("--with_reconstruction", action="store_true", help='If exist use the model with reconstruction \
of the original image during training and use additional comparison loss of the original and reconstructed image.')
    parser.add_argument("--run_without_training", action="store_true", help='Run framework without invoking training call. \
All other functionalities still work the same. Running with "disable_dreams_gen" will run only test.') ##**
    parser.add_argument("--disable_shuffle", action="store_false", help='Flag to shuffle train normal and dream datasets. If \
flag "disable_dream_shuffle" is set then it takes precedence over this flag.')
    parser.add_argument("--datasampler_type", type=str, default='none', help='''Select datasampler type.
Avaliable types are: 
"none" - no datasampler.
"v2" - datasampler where you can choose division into the majority class and other class in single batch.''')
    parser.add_argument("--cyclic_latent_buffer_size_per_class", type=int, default=40, help='Size per class of the cyclic \
buffer used in island overlay for model.')
    parser.add_argument("--train_with_logits", action="store_true", help='Use dataset with logits.')
    parser.add_argument("--num_workers", type=int, default=4, help='Number of dataloader workers.')
    parser.add_argument("--dream_num_workers", type=int, default=0, help='Number of dream dataloader workers.')
    parser.add_argument("--test_val_num_workers", type=int, default=4, help='Number of test and validation dataloader workers.')
    parser.add_argument("--save_trained_model", action="store_true", help='') ##**
    parser.add_argument("--save_model_name", type=str, help='') ##**
    parser.add_argument("--load_model", type=str, help='') ##**
    parser.add_argument("--enable_checkpoint", action="store_true", help='')
    parser.add_argument("--optimizer_type", type=str, default='adam', help='')
    parser.add_argument("--scheduler_type", type=str, default='none', help='Type of scheduler. Use "train_scheduler_steps" \
to choose epoch at which to call it.')
    parser.add_argument("--reset_optim_type", type=str, default='default', help='')
    parser.add_argument("--export_path", type=str, help='')
    parser.add_argument("--save_dreams", type=str, help='') ##**
    parser.add_argument("--load_dreams", type=str, help='') ##**
    
    ######################################
    #####    numerical parameters   ######
    ######################################
    parser.add_argument("--batch_size", type=int, default=32, help='Size of the batch.')
    parser.add_argument("--num_tasks", type=int, default=1, help='How many tasks will be created')
    parser.add_argument("--num_loops", type=int, default=1, help='How many loops will be traversed. \
Each new loop will increment current task index. If "num_loops">"num_tasks" then at each new loop \
the last task in array will be used.') ##**
    parser.add_argument("--epochs_per_task", type=int, default=5, help='How many epochs do per one task in "num_tasks"') ##**
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning rate of the optimizer.')
    parser.add_argument("--gamma", type=float, default=1, help='Gamma parameter for optimizer if exist.')
    parser.add_argument("--norm_lambda", type=float, default=0., help='Lambda parametr of the used l2 normalization. If 0. then \
no normalization is used. Normalization is used to the last model layer, the latent output of the "CLModelWithIslands".')
    parser.add_argument("--train_scheduler_steps", nargs='+', type=int, default=(3, ), help='Epoch training steps \
at where to call scheduler, change learning rate. Use "scheduler_type" to enable scheduler.')
    parser.add_argument("--number_of_classes", type=int, default=10, help='Number of classes model should output. \
If less than in dataset then model will be trained and validated only using this number of classes')

    ######################################
    #####     dream parameters      ######
    ######################################
    parser.add_argument("--dream_obj_type", nargs='+', type=str, help='Model objective funtions type. May be multiple')
    parser.add_argument("--dreams_per_target", type=int, default=128, help='How many epochs do per one task in "num_tasks"')
    parser.add_argument("--dreaming_batch_size", type=int, default=128, help='How many images \
in batch during dreaming should be produced.')
    parser.add_argument("--dream_lr", type=float, default=1e-3, help='Learning rate of the dream optimizer.')
    parser.add_argument("--dream_threshold", nargs='+', type=int, default=[1024, ], help='How many iterations should \
be used to generate an output image during dreaming, using only max value. Values lesser than max are points where the \
images from the batch will be additionaly saved.')
    parser.add_argument("--dream_frequency", type=int, default=1, help='How often dream images should be used during \
training. The bigger value the lesser frequency.')
    parser.add_argument("--disable_dreams_gen", action="store_false", help='If framework should produce dreams and use them \
during training. Running with "run_without_training" will run only test.') ##**
    parser.add_argument("--enable_dream_transforms", action="store_false", help='Enable and add all default \
tranforms on dreamed images used in lucid framework in main function.') ##**
    parser.add_argument("--disable_dream_shuffle", action="store_false", help='Flag to shuffle only train dream dataset')
    parser.add_argument("--param_image", type=str, default='image', help='Type of image. Default \
"image"-normal image  "cppn"-cppn image (does not use "dreaming_batch_size")')
    parser.add_argument("--train_only_dream_batch", action="store_true", help='Use this flag to train only on dream batch \
after first epoch when dream batch is created.') ##**
    parser.add_argument("--generate_dreams_at_start", action="store_true", help='Start generating dreams from the start of\
the first epoch. This can be used with flag "run_without_training" to only generate dreams from loaded model.')


    ######################################
    #####       dataset swap        ######
    ######################################
    parser.add_argument("--swap_datasets", action="store_true", help='Do training once, generate dream dataset and then\
run training on a newly initialized model using only dream dataset. Not compatible with "dream_frequency", \
"train_only_dream_batch". Odd task number will indicate normal training and even task number will indicate dream training.') ##**
    parser.add_argument("--reload_model_after_loop", action="store_true", help='Reload model weights after each main loop.\
Reloading means any weights that model had before training will be reloaded.') ##**
    parser.add_argument("--reinit_model_after_loop", action="store_true", help='Reset model after each main loop.\
Model will have newly initialized weights after each main loop.') ##**
    parser.add_argument("--weight_reset_sanity_check", action="store_true", help='Enable sanity check for reload/reinit weights.')



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
    #####     hooking to model      ######
    ######################################
    parser.add_argument("--use_layer_loss_at", type=int, help='Use layer loss function \
at given fit loop index. Default None means this functionality is not enabled. Value less than zero \
means it will be used just like the python indexing for negative numbers.')


    ######################################
    ######          other           ######
    ######################################
    parser.add_argument("--run_config_folder", type=str, default='run_conf', 
        help='Folder where to save and load argparse config for flags "config" and "config_export" ')


    args = parser.parse_args()

    args = load_config(args, parser)
    export_config(args)
    
    return args, parser

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
        'lr': args.lr,
        'gamma':args.gamma,
    }

def log_to_wandb(args):
    #if(wandb.run is not None): # does not work
    wandb.config.update({'Plain args': str(sys.argv)})
    wandb.config.update(args)
    print('Input command line:')
    print(' '.join(sys.argv))
    print('Used config:')
    print(wandb.config)

def load_config(args: Namespace, parser: ArgumentParser) -> Namespace:
    if args.config is not None:
        for conf_fname in args.config:
            folder = Path(args.run_config_folder)
            file = folder / conf_fname
            print(file)
            file = glob.glob(str(file), recursive=True)
            if(len(file) != 1):
                raise Exception(f'Cannot load dreams - no or too many matching filenames. From "{conf_fname}" found only these paths: {file}')
            file = file[0]
            with open(file, 'r') as f:
                parser.set_defaults(**json.load(f))
        
        args = parser.parse_args()
    return args

def export_config(args: Namespace) -> None:
    if args.config_export and not args.fast_dev_run:
        tmp_args = vars(args).copy()
        del tmp_args['config_export']  # Do not dump value of conf_export flag
        del tmp_args['config']  # Values already loaded
        del tmp_args['fast_dev_run']  # Fast dev run should not be present in config file

        # Remove all options that are corelated to saving and loading
        del tmp_args['load_dreams']  
        del tmp_args['save_dreams']  
        del tmp_args['export_path'] 
        del tmp_args['save_model_name'] 
        del tmp_args['load_model'] 
        del tmp_args['save_trained_model'] 
        del tmp_args['enable_checkpoint'] 
        path = Path(args.run_config_folder) / args.config_export
        Path.mkdir(path.parent, parents=True, exist_ok=True)
        dump = json.dumps(tmp_args,  indent=4, sort_keys=True)
        with open(path, 'w') as f:
            f.write(dump)