from latent_dreams import logic
import my_parser
import gc, sys
import torch
import shlex
import traceback
from datetime import datetime
from argparse import ArgumentParser, Namespace
import wandb
import time
from collections.abc import Sequence

def main():
    """
        Change fast_dev_run flag to run full experiments.
    """
    if __name__ == "__main__":
        parser = ArgumentParser(prog='Continual dreaming', add_help=True, description='Main experiments')
        parser.add_argument("-f", "--fast_dev_run", action="store_true", help='Use to fast check for errors in code.') ##**
        parser.add_argument("-r", "--repeat", type=int, default=1 , help='How many times repeat experiments.') ##**
        parser.add_argument("--project_name", type=str, default=None , help='Name of the project. If None then it will be generated.') ##**        
        args = parser.parse_args()


        parser_exp = my_parser.main_arg_parser()
        parser_exp = my_parser.layer_statistics_arg_parser(parser_exp)
        parser_exp = my_parser.dream_parameters_arg_parser(parser_exp)
        parser_exp = my_parser.model_statistics_optim_arg_parser(parser_exp)

        today_time = datetime.today().strftime('%Y-%m-%d=%H-%M-%S')

        if(len(grid_search_dict) != 0):
            input_commands = grid_search_numerical(chi_sqr_c100_sgd_search_tmpl, "chi_sqr_c100_sgd_search", grid_search_dict)
            experiments.update(input_commands)

        print(f"Experiments to be run:")
        for v in experiments.values():
            print(f"* {v}")

        print()
        for k, v in experiments.items():
            for idx in range(1, args.repeat + 1):
                print(f"Running experiment: {k}; repeat {idx}/{args.repeat}")

                v = v.replace('\n', ' ')
                if(args.fast_dev_run):
                    v += ' -f'
                args_exp = my_parser.parse_args(parser_exp, shlex.split(v))
                try:
                    project_name = args.project_name
                    if(args.project_name is None):
                        project_name = f"exp_{today_time}"
                    
                    logic(args_exp, True, project_name=project_name, run_name=k)
                except KeyboardInterrupt:
                    print("Experiment KeyboardInterrupt occurred")
                    print("Experiment Exception occurred")
                    print(traceback.format_exc())
                    print("Sleep 10 seconds...")
                    sys.stdout.flush()
                    time.sleep(10) # wait for a while for wandb to send all logs to server.
                    wandb.finish()
                    exit()
                except Exception:
                    print("Sleep 10 seconds...")
                    print("Experiment Exception occurred")
                    print(traceback.format_exc())
                    sys.stdout.flush()
                    time.sleep(10)
                    wandb.finish()

                print(f"End of experiment: {k}; repeat {idx}/{args.repeat}")
                print("Clearing gpu cache and invoking garbage collector")

                torch.cuda.empty_cache()
                gc.collect()
                print("Done")

def grid_search_recursive(input: str, search_args: dict, exp_name: str, key_idx: int, ret: dict[str, str]) -> None:
    if(key_idx >= len(search_args)):
        ret[exp_name] = input
        return 
    key = list(search_args.keys())[key_idx]

    if not (key in search_args):
        raise Exception(f"Wrong key parameter: {key}")
    if isinstance(search_args[key], Sequence) and len(search_args[key]) == 4:
        name, start, stop, step = search_args[key]

        for value in range(start, stop, step):
            grid_search_recursive_call(input=input, key=key, value=value, exp_name=exp_name, name=name, search_args=search_args, key_idx=key_idx, ret=ret)

    elif isinstance(search_args[key], Sequence) and len(search_args[key]) == 2:
        name, value = search_args[key]
        grid_search_recursive_call(input=input, key=key, value=value, exp_name=exp_name, name=name, search_args=search_args, key_idx=key_idx, ret=ret)
    else:
        raise Exception(f"Wrong value parameter: {key}: {search_args[key]}")

def grid_search_recursive_call(input, key, value, exp_name, name, search_args, key_idx, ret):
    new_input = f'{input} {key} {value}'
    new_exp_name = f"{exp_name}_{name}_{value}"
    grid_search_recursive(new_input, search_args, new_exp_name, key_idx + 1, ret=ret)

def grid_search_numerical(input: str, exp_name, search_args: dict[tuple[str, str], tuple[float, float, float]]) -> dict[str, str]:
    ret = {}
    grid_search_recursive(input, search_args, exp_name, 0, ret=ret)
    return ret

crossentropy_default_c10_sgd = """
-d c10 --model.num_classes 10 --loop.schedule 200 \
--config.framework_type crossentropy-default --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 100 150 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --config.seed 2024 \
"""

crossentropy_default_c100_sgd = """
-d c100 --model.num_classes 100 --loop.schedule 200 \
--config.framework_type crossentropy-default --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 100 150 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --config.seed 2024 \
"""

chi_sqr_c10_sgd = """
-d c10 --model.num_classes 10 --loop.schedule 260 \
--config.framework_type latent-multitarget --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --model.latent.size 3 --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 10 \
--model.loss.chi.scale 30 --model.loss.chi.ratio_gamma 2 --datamodule.batch_size 32 \
--model.loss.chi.ratio_milestones 5 20 40 60 --config.seed 2024 \
"""

chi_sqr_c100_sgd = """
-d c100 --model.num_classes 100 --loop.schedule 260 \
--config.framework_type latent-multitarget --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --model.latent.size 30 --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 10 \
--model.loss.chi.scale 100 --model.loss.chi.ratio_gamma 2 --datamodule.batch_size 320 \
--model.loss.chi.ratio_milestones 5 20 40 60 --config.seed 2024 \
"""

chi_sqr_c100_sgd_search_tmpl = """
-d c100 --model.num_classes 100 --loop.schedule 300 \
--config.framework_type latent-multitarget --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio_gamma 2  \
--model.loss.chi.ratio_milestones 40 60 80 100 --config.seed 2024 \
"""



experiments = {
    "crossentropy_default_c10_sgd": crossentropy_default_c10_sgd,
    "crossentropy_default_c100_sgd": crossentropy_default_c10_sgd,
    "chi_sqr_c10_sgd": chi_sqr_c10_sgd,
    "chi_sqr_c100_sgd": chi_sqr_c100_sgd,
}

grid_search_dict = {
    "--model.latent.size": ["latent_size", 3, 12, 2],
    "--model.loss.chi.ratio": ["chi_ratio", 10],
    "--model.loss.chi.scale": ["chi_scale", 80, 181, 20],
    "--datamodule.batch_size": ["batch_size", 120, 321, 50]
}


main()