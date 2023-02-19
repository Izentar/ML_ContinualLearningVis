from argparse import ArgumentParser
import wandb
import sys

def arg_parser():
    parser = ArgumentParser(prog='Continual dreaming', add_help=True, description='Configurable framework to work with\
continual learning tasks.', 
epilog="""
Validation dataset uses data from test dataset but divided into appropriate tasks.
Test dataset uses test data with only the classes used in previous tasks. 
""")
    
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("-d", "--dataset", type=str)
    parser.add_argument("--early_finish_at", type=int, default=-1, help='Finish training loop at desired epoch. Default "-1"')
    parser.add_argument("--with_reconstruction", action="store_true", help='If exist use the model with reconstruction \
of the original image during training and use additional comparison loss of the original and reconstructed image.')
    parser.add_argument("--run_without_training", action="store_true", help='Run framework without invoking training call. \
All other functionalities still work the same.')
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
    parser.add_argument("--dream_num_workers", type=int, default=4, help='Number of dream dataloader workers.')
    parser.add_argument("--test_val_num_workers", type=int, default=4, help='Number of test and validation dataloader workers.')
    parser.add_argument("--save_trained_model", type=str, help='')
    parser.add_argument("--load_model", type=str, help='')
    parser.add_argument("--enable_checkpoint", action="store_true", help='')
    parser.add_argument("--disable_normal_dataset", action="store_true", help='Do not use normal dataset, only dream dataset.')
    parser.add_argument("--optimizer_type", type=str, default='adam', help='')
    parser.add_argument("--scheduler_type", type=str, default='none', help='')
    parser.add_argument("--reset_optim_type", type=str, default='default', help='')
    
    ######################################
    #####    numerical parameters   ######
    ######################################
    parser.add_argument("--batch_size", type=int, default=32, help='Size of the batch.')
    parser.add_argument("--num_tasks", type=int, default=1, help='How many tasks will be created')
    parser.add_argument("--num_loops", type=int, default=1, help='How many loops will be traversed. \
Each new loop will increment current task index. If "num_loops">"num_tasks" then at each new loop \
the last task in array will be used.')
    parser.add_argument("--epochs_per_task", type=int, default=5, help='How many epochs do per one task in "num_tasks"')
    parser.add_argument("--lr", type=float, default=1e-3, help='Learning rate of the optimizer.')
    parser.add_argument("--norm_lambda", type=float, default=0., help='Lambda parametr of the used l2 normalization. If 0. then \
no normalization is used. Normalization is used to the last model layer, the latent output of the "CLModelWithIslands".')
    parser.add_argument("--train_scheduler_steps", nargs='+', type=int, default=(3, ), help='Epoch training steps \
at where to call scheduler, change learning rate')
    parser.add_argument("--number_of_classes", type=int, default=10, help='Number of classes model should output. \
If less than in dataset then model will be trained and validated only using this number of classes')
    parser.add_argument("--gamma", type=float, default=1, help='')

    ######################################
    #####     dream parameters      ######
    ######################################
    parser.add_argument("--dreams_per_target", type=int, default=64, help='How many epochs do per one task in "num_tasks"')
    parser.add_argument("--target_images_per_dreaming_batch", type=int, default=8, help='How many images \
in batch during dreaming should be produced.')
    parser.add_argument("--dream_threshold", nargs='+', type=int, default=(1024, ), help='How many iterations should \
be used to generate an output image during dreaming, using only max value. Values lesser than max are points where the \
images from the batch will be additionaly saved.')
    parser.add_argument("--dream_frequency", type=int, default=1, help='How often dream images should be used during \
training. The bigger value the lesser frequency.')
    parser.add_argument("--disable_dreams", action="store_false", help='If framework should produce dreams and use them \
during training.')
    parser.add_argument("--enable_dream_transforms", action="store_false", help='Enable and add all default \
tranforms on dreamed images used in lucid framework in main function.')
    parser.add_argument("--disable_dream_shuffle", action="store_false", help='Flag to shuffle only train dream dataset')
    parser.add_argument("--param_image", type=str, default='image', help='Type of image. Default \
"image"-normal image  "cppn"-cppn image')
    parser.add_argument("--train_only_dream_batch", action="store_true", help='Use this flag to train only on dream batch \
after first epoch when dream batch is created.')


    ######################################
    #####       dataset swap        ######
    ######################################
    parser.add_argument("--swap_datasets", action="store_true", help='Do training once, generate dream dataset and then\
run training on a newly initialized model using only dream dataset. Not compatible with "dream_frequency", \
"train_only_dream_batch". Odd task number will indicate normal training and even task number will indicate dream training.')
    parser.add_argument("--reload_model_after_loop", action="store_true", help='Reload model weights after each main loop.\
Reloading means any weights that model had before training will be reloaded.')
    parser.add_argument("--reinit_model_after_loop", action="store_true", help='Reset model after each main loop.\
Model will have newly initialized weights after each main loop.')
    parser.add_argument("--weight_reset_sanity_check", action="store_true", help='Enable sanity check for reload/reinit weights.')



    ######################################
    #####       fast dev run        ######
    ######################################
    parser.add_argument("-f", "--fast-dev-run", action="store_true", help='Use to fast check for errors in code.\
It ')
    parser.add_argument("--fast_dev_run_batches", type=int, default=30, help='')
    parser.add_argument("--fast_dev_run_epochs", type=int, default=1, help='')
    parser.add_argument("--fast_dev_run_dream_threshold", nargs='+', type=int, default=(32, ), help='')


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

    
    return parser.parse_args()

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
    wandb.init()
    wandb.config.update({'Plain args': str(sys.argv)})
    wandb.config.update(args)
    print('Input command line:')
    print(' '.join(sys.argv))
    print('Used config:')
    print(wandb.config)