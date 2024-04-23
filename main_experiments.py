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
        parser = ArgumentParser(prog='Continual dreaming', add_help=True, description='Main experiments. Change this script at wish.')
        parser.add_argument("-f", "--fast_dev_run", action="store_true", help='Use to fast_dev_run check for errors in code.') ##**
        parser.add_argument("-nl", "--nologic", action="store_true" , help='Run without invoking script logic.') ##**
        parser.add_argument("-r", "--repeat", type=int, default=1 , help='How many times repeat experiments.') ##**
        parser.add_argument("--project_name", type=str, default=None , help='Name of the project. If None then it will be generated.') ##**        
        parser.add_argument("--start_at", type=int, default=1, help='Index of the experiment to start with \
(experiments sorted as: normal first; grid search second). Index starts at 1.') ##**       
        parser.add_argument("--end_at", type=int, default=-1, help='Index of the experiment to end with \
(experiments sorted as: normal first; grid search second). Index starts at 1.') ##**   
        args = parser.parse_args()


        parser_exp = my_parser.main_arg_parser()
        parser_exp = my_parser.layer_statistics_arg_parser(parser_exp)
        parser_exp = my_parser.dream_parameters_arg_parser(parser_exp)
        parser_exp = my_parser.model_statistics_optim_arg_parser(parser_exp)

        today_time = datetime.today().strftime('%Y-%m-%d=%H-%M-%S')

        if(len(grid_search_dict) != 0):
            input_commands = grid_search_numerical(exp_template, "exp_search", grid_search_dict)
            experiments.update(input_commands)

        print(f"Experiments to be run:")
        for v in experiments.values():
            print(f"* {v}")

        print()
        loop_counter = 0
        for k, v in experiments.items():
            for idx in range(1, args.repeat + 1):
                loop_counter += 1
                print(f"Running experiment: {k}; repeat {idx}/{args.repeat}")
                if(args.start_at > loop_counter):
                    print(f"Experiment skipped because of 'start_at' argument ({args.start_at} > {loop_counter}).")
                    continue
                if(args.end_at != -1 and args.end_at <= loop_counter):
                    print(f"Experiment ended because of 'end_at' argument ({args.end_at} <= {loop_counter}).")
                    exit()

                v = v.replace('\n', ' ')
                if(args.fast_dev_run):
                    v += ' -f'
                    full_command = shlex.split(v)
                args_exp = my_parser.parse_args(parser_exp, full_command)
                try:
                    project_name = args.project_name
                    if(args.project_name is None):
                        project_name = f"exp_{today_time}"
                    
                    if(not args.nologic):
                        logic(args_exp, True, project_name=project_name, run_name=k, full_text_command=' '.join(full_command))
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
    if isinstance(search_args[key], Sequence) and len(search_args[key]) == 2 and isinstance(search_args[key][1], Sequence) \
            and not isinstance(search_args[key][1], str):
        # type [name, list of values]
        # but not a string
        name, range_list = search_args[key]
        for value in range_list:
            grid_search_recursive_call(input=input, key=key, value=value, exp_name=exp_name, name=name, search_args=search_args, key_idx=key_idx, ret=ret)
    
    elif isinstance(search_args[key], Sequence) and len(search_args[key]) == 4:
        # type name, start, stop, step
        name, start, stop, step = search_args[key]

        for value in range(start, stop + 1, step):
            grid_search_recursive_call(input=input, key=key, value=value, exp_name=exp_name, name=name, search_args=search_args, key_idx=key_idx, ret=ret)

    elif isinstance(search_args[key], Sequence) and len(search_args[key]) == 2:
        # type name, single value
        name, value = search_args[key]
        grid_search_recursive_call(input=input, key=key, value=value, exp_name=exp_name, name=name, search_args=search_args, key_idx=key_idx, ret=ret)
    else:
        raise Exception(f"Wrong value parameter: {key}: {search_args[key]}")

def grid_search_recursive_call(input, key, value, exp_name, name, search_args, key_idx, ret):
    if(isinstance(value, Sequence)) and not isinstance(value, str):
        value_new = ""
        for v in value:
            value_new += f" {v} "
        value = value_new
    new_input = f'{input} {key} {value}'
    if(len(name) != 0):
        new_exp_name = f"{exp_name}_{name}_{value}"
    else:
        new_exp_name = exp_name
    grid_search_recursive(new_input, search_args, new_exp_name, key_idx + 1, ret=ret)

def grid_search_numerical(input: str, exp_name, search_args: dict[tuple[str, str], tuple[float, float, float]]) -> dict[str, str]:
    ret = {}
    grid_search_recursive(input, search_args, exp_name, 0, ret=ret)
    return ret


chi_sqr_c10_sgd = """
-d c10 --model.num_classes 10 --loop.schedule 260 \
--config.framework_type latent-multitarget --model.type \
DLA --loop.num_loops 1 --loop.train_at 0 \
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
DLA --loop.num_loops 1 --loop.train_at 0 \
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
--config.framework_type latent-multitarget --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio_gamma 2  \
--model.loss.chi.ratio_milestones 40 60 80 100 --config.seed 2024 \
"""

chi_sqr_continual_learning_search_tmpl = """
-d c100 --model.num_classes 100 --model.latent.size 10 --config.num_tasks 2 --loop.schedule 300 300 0 \
--config.framework_type latent-multitarget-multitask \
--loop.num_loops 3 --loop.train_at 0 1 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable --stat.collect_stats.use_dream_dataset  \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio_gamma 2  \
--model.loss.chi.ratio_milestones 40 60 100 --config.seed 2024 \
--model.loss.chi.ratio 10 --model.loss.chi.scale 120 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 2 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at True --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 440 --loop.vis.generate_at 1 2 3 --datamodule.vis.standard_image_size 32 \
"""

#########################
#########################
#########################



#########################

# change model at wish
crossentropy_default_c10_sgd_tmpl = """
-d c10 --model.num_classes 10 --loop.schedule 200 \
--config.framework_type crossentropy-default --model.type \
DLA --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 100 150 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --config.seed 2024 \
"""

# change model at wish
crossentropy_default_c100_sgd_tmpl = """
-d c100 --model.num_classes 100 --loop.schedule 200 \
--config.framework_type crossentropy-default --model.type \
DLA --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 100 150 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --config.seed 2024 \
"""

chi_sqr_train_full_vgg_tmpl = """
-d c100 --model.num_classes 100 --loop.schedule 300 \
--config.framework_type latent-multitarget --model.type \
VGG --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio_gamma 2  \
--model.loss.chi.ratio_milestones 40 60 80 100 --config.seed 2024 \
"""

chi_sqr_train_full_resnet_tmpl = """
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

chi_sqr_train_full_dla_tmpl = """
-d c100 --model.num_classes 100 --loop.schedule 300 \
--config.framework_type latent-multitarget --model.type \
DLA --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio_gamma 2  \
--model.loss.chi.ratio_milestones 40 60 80 100 --config.seed 2024 \
"""

#########################################################################

chi_sqr_sgd_train_full_const_ratio_tmpl = """
-d c100 --model.num_classes 100 --config.num_tasks 1 --loop.schedule 300 0 --model.latent.size 10 \
--config.framework_type latent-multitarget-multitask \
--loop.num_loops 2 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable  \
--model.loss.chi.shift_min_distance 0 --config.seed 2024 \
--model.loss.chi.ratio 10 --model.loss.chi.scale 120 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at False --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 220 --loop.vis.generate_at 1 --datamodule.vis.standard_image_size 32 \
--loop.vis.image_reg.var.scale 0.001 --loop.vis.layerloss.deep_inversion.scale 10 --datamodule.vis.optim.kwargs.lr 0.05
"""

chi_sqr_sgd_train_full_const_ratio_c10_tmpl = """
-d c10 --model.num_classes 10 --config.num_tasks 1 --loop.schedule 300 0 --model.latent.size 3 \
--config.framework_type latent-multitarget-multitask \
--loop.num_loops 2 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable  \
--model.loss.chi.shift_min_distance 0 --config.seed 2024 \
--model.loss.chi.ratio 10 --model.loss.chi.scale 120 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at False --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 220 --loop.vis.generate_at 1 --datamodule.vis.standard_image_size 32 \
--loop.vis.image_reg.var.scale 0.001 --loop.vis.layerloss.deep_inversion.scale 10 --datamodule.vis.optim.kwargs.lr 0.05
"""


cross_entropy_sgd_train_full_and_vis_tmpl = """
-d c100 --model.num_classes 100 --config.num_tasks 1 --loop.schedule 300 0 \
--config.framework_type crossentropy-multitarget-multitask \
--loop.num_loops 2 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable --config.seed 2024 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at False --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 220 --loop.vis.generate_at 1 --datamodule.vis.standard_image_size 32 \
--loop.vis.image_reg.var.scale 0.001 \
--loop.vis.layerloss.deep_inversion.scale 10 --datamodule.vis.optim.kwargs.lr 0.05
"""

cross_entropy_sgd_train_full_and_vis_c10_tmpl = """
-d c10 --model.num_classes 10 --config.num_tasks 1 --loop.schedule 300 0 \
--config.framework_type crossentropy-multitarget-multitask \
--loop.num_loops 2 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable --config.seed 2024 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at False --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 220 --loop.vis.generate_at 1 --datamodule.vis.standard_image_size 32 \
--loop.vis.image_reg.var.scale 0.001 \
--loop.vis.layerloss.deep_inversion.scale 10 --datamodule.vis.optim.kwargs.lr 0.05
"""

chi_sqr_sgd_train_full_and_vis_tmpl = """
-d c100 --model.num_classes 100 --config.num_tasks 1 --loop.schedule 300 0 \
--config.framework_type latent-multitarget-multitask \
--loop.num_loops 2 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable  \
--model.loss.chi.shift_min_distance 0 --config.seed 2024 \
--model.loss.chi.ratio 10 --model.loss.chi.scale 120 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at False --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 220 --loop.vis.generate_at 1 --datamodule.vis.standard_image_size 32 \
--loop.vis.image_reg.var.scale 0.001 --model.loss.chi.ratio_milestones 40 60 100 \
--loop.vis.layerloss.deep_inversion.scale 10 --datamodule.vis.optim.kwargs.lr 0.05
"""

chi_sqr_sgd_train_full_and_vis_c10_tmpl = """
-d c10 --model.num_classes 10 --config.num_tasks 1 --loop.schedule 300 0 \
--config.framework_type latent-multitarget-multitask \
--loop.num_loops 2 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable  \
--model.loss.chi.shift_min_distance 0 --config.seed 2024 \
--model.loss.chi.ratio 10 --model.loss.chi.scale 120 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at False --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 220 --loop.vis.generate_at 1 --datamodule.vis.standard_image_size 32 \
--loop.vis.image_reg.var.scale 0.001 --model.loss.chi.ratio_milestones 40 60 100 \
--loop.vis.layerloss.deep_inversion.scale 10 --datamodule.vis.optim.kwargs.lr 0.05
"""

cross_entropy_sgd_train_full_and_vis_multitask_tmpl = """
-d c100 --model.num_classes 100 --config.num_tasks 2 --loop.schedule 300 300 0 \
--config.framework_type crossentropy-multitarget-multitask \
--loop.num_loops 3 --loop.train_at True  \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable --config.seed 2024 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 2 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at False --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 440 --loop.vis.generate_at 1 2 --datamodule.vis.standard_image_size 32 \
--loop.vis.image_reg.var.scale 0.001 \
--loop.vis.layerloss.deep_inversion.scale 10 --datamodule.vis.optim.kwargs.lr 0.05
"""

cross_entropy_sgd_train_full_and_vis_multitask_c10_tmpl = """
-d c10 --model.num_classes 10 --config.num_tasks 2 --loop.schedule 300 300 0 \
--config.framework_type crossentropy-multitarget-multitask \
--loop.num_loops 3 --loop.train_at True  \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable --config.seed 2024 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 2 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at False --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 440 --loop.vis.generate_at 1 2 --datamodule.vis.standard_image_size 32 \
--loop.vis.image_reg.var.scale 0.001 \
--loop.vis.layerloss.deep_inversion.scale 10 --datamodule.vis.optim.kwargs.lr 0.05
"""

chi_sqr_sgd_train_full_and_vis_multitask_tmpl = """
-d c100 --model.num_classes 100 --config.num_tasks 2 --loop.schedule 300 300 0 \
--config.framework_type latent-multitarget-multitask \
--loop.num_loops 3 --loop.train_at True \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable  \
--model.loss.chi.shift_min_distance 0 --config.seed 2024 \
--model.loss.chi.ratio 10 --model.loss.chi.scale 120 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 2 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at False --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 440 --loop.vis.generate_at 1 2 --datamodule.vis.standard_image_size 32 \
--loop.vis.image_reg.var.scale 0.001 --model.loss.chi.ratio_milestones 40 60 100 \
--loop.vis.layerloss.deep_inversion.scale 10 --datamodule.vis.optim.kwargs.lr 0.05
"""

chi_sqr_sgd_train_full_and_vis_multitask_c10_tmpl = """
-d c10 --model.num_classes 10 --config.num_tasks 2 --loop.schedule 300 300 0 \
--config.framework_type latent-multitarget-multitask \
--loop.num_loops 3 --loop.train_at True \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root model_save/test \
--stat.collect_stats.enable  \
--model.loss.chi.shift_min_distance 0 --config.seed 2024 \
--model.loss.chi.ratio 10 --model.loss.chi.scale 120 --datamodule.batch_size 220 \
--datamodule.vis.only_vis_at False --datamodule.vis.enable_vis_at 1 2 --loop.vis.image_reg.var.use_at True \
--loop.vis.image_reg.l2.use_at False --loop.test_at True \
--loop.vis.layerloss.deep_inversion.use_at True --datamodule.vis.optim.type adam  \
--datamodule.vis.image_type pixel --datamodule.vis.threshold 500 \
--loop.save.dreams --datamodule.vis.multitarget.enable --datamodule.vis.batch_size 220 \
--datamodule.vis.per_target 440 --loop.vis.generate_at 1 2 --datamodule.vis.standard_image_size 32 \
--loop.vis.image_reg.var.scale 0.001 --model.loss.chi.ratio_milestones 40 60 100 \
--loop.vis.layerloss.deep_inversion.scale 10 --datamodule.vis.optim.kwargs.lr 0.05
"""


experiments = {
    #"crossentropy_default_c10_sgd_tmpl": crossentropy_default_c10_sgd_tmpl,
    #"crossentropy_default_c100_sgd_tmpl": crossentropy_default_c100_sgd_tmpl,
    #"chi_sqr_c10_sgd": chi_sqr_c10_sgd,
    #"chi_sqr_c100_sgd": chi_sqr_c100_sgd,
}

# for note:
# it can process 3 types of input:
# - (single value)
# - (start, stop, step)
# - ([list of values])

#grid_search_dict = {
#    "--model.latent.size": ["latent_size", [3, 10, 20, 30]],
#    "--model.loss.chi.ratio": ["chi_ratio", 10],
#    "--model.loss.chi.scale": ["chi_scale", 80, 160, 40],
#    "--datamodule.batch_size": ["batch_size", 120, 320, 100]
#}

#grid_search_dict = {
#    "--model.latent.size": ["latent_size", [20, 30]],
#    "--model.loss.chi.ratio": ["chi_ratio", 10],
#    "--model.loss.chi.scale": ["chi_scale", 80, 160, 40],
#    "--datamodule.batch_size": ["batch_size", 120, 320, 100],
#    "--model.type": ["model_type", ["DLA", "VGG", "custom-resnet34"]],
#}


grid_search_normal_train_dict = {
    "--model.latent.size": ["latent_size", 10],
    "--model.loss.chi.ratio": ["chi_ratio", 10],
    "--model.loss.chi.scale": ["chi_scale", 160],
    "--datamodule.batch_size": ["batch_size", 220],
    "--model.type": ["model_type", "DLA"],
    "--model.sched.kwargs.milestones": ["lr_milestones", [[40, 60, 80, 100], [60, 100], [40, 60, 80, 100, 120, 140]]],
}

grid_search_continual_learning_resnet_dict = {
    #"--loop.load.name": ["", "/home/ubuntu/models/cky76ok9/checkpoints/epoch=299-step=68400.ckpt"],
    "--model.type": ["model_type", ["custom-resnet34"]],
    "--loop.vis.image_reg.var.scale": ["vis_var_scale", [0.01]],
    "--loop.vis.image_reg.l2.coeff": ["vis_l2_coeff", [1e-05]],
    "--loop.vis.layerloss.deep_inversion.scale": ["deep_inv_scale", [0.1]],
    "--datamodule.vis.optim.kwargs.lr": ["vis_lr", [0.05]],
}

grid_search_continual_learning_vgg_dict = {
    #"--loop.load.name": ["", "/home/ubuntu/models/..."],
    "--model.type": ["model_type", ["vgg"]],
    "--loop.vis.image_reg.var.scale": ["vis_var_scale", [0.01]],
    "--loop.vis.image_reg.l2.coeff": ["vis_l2_coeff", [1e-05]],
    "--loop.vis.layerloss.deep_inversion.scale": ["deep_inv_scale", [0.1]],
    "--datamodule.vis.optim.kwargs.lr": ["vis_lr", [0.05]],
}

grid_search_continual_learning_dla_dict = {
    #"--loop.load.name": ["", "/home/ubuntu/models/..."],
    "--model.type": ["model_type", ["dla"]],
    "--loop.vis.image_reg.var.scale": ["vis_var_scale", [0.01]],
    "--loop.vis.image_reg.l2.coeff": ["vis_l2_coeff", [1e-05]],
    "--loop.vis.layerloss.deep_inversion.scale": ["deep_inv_scale", [0.1]],
    "--datamodule.vis.optim.kwargs.lr": ["vis_lr", [0.05]],
}

#chi_sqr_sgd_train_full_and_vis_grid_search = {
#    "--model.type": ["model_type", ["dla", "vgg", "custom-resnet34"]],
#    "--model.latent.size": ["latent_size", [3, 10, 20, 30]],
#}

#########################
#########################
#########################

chi_sqr_train_full_vgg_resnet_dla_grid_search = {
    "--model.latent.size": ["latent_size", [3, 10, 20, 30]],
    "--model.loss.chi.ratio": ["chi_ratio", 10],
    "--model.loss.chi.scale": ["chi_scale", 80, 160, 40],
    "--datamodule.batch_size": ["batch_size", 120, 320, 100]
}

cross_entropy_sgd_train_full_and_vis_grid_search = {
    "--model.type": ["model_type", ["dla", "vgg", "custom-resnet34"]],
}


chi_sqr_sgd_train_full_and_vis_grid_search = {
    "--model.type": ["model_type", ["dla", "vgg", "custom-resnet34"]],
    "--model.latent.size": ["latent_size", 10],
}

chi_sqr_sgd_train_full_and_vis_c10_grid_search = {
    "--model.type": ["model_type", ["dla", "vgg", "custom-resnet34"]],
    "--model.latent.size": ["latent_size", 3],
}

cross_entropy_sgd_train_full_and_vis_multitask_grid_search = {
    "--model.type": ["model_type", ["dla", "vgg", "custom-resnet34"]],
}

chi_sqr_sgd_train_full_and_vis_multitask_grid_search = {
    "--model.type": ["model_type", ["dla", "vgg", "custom-resnet34"]],
    "--model.latent.size": ["latent_size", 10],
}

chi_sqr_sgd_train_full_and_vis_multitask_c10_grid_search = {
    "--model.type": ["model_type", ["dla", "vgg", "custom-resnet34"]],
    "--model.latent.size": ["latent_size", 3],
}

# experiment for chi2, only train and full grid search
#grid_search_dict = chi_sqr_train_full_vgg_resnet_dla_grid_search
#exp_template = chi_sqr_train_full_vgg_tmpl
#exp_template = chi_sqr_train_full_resnet_tmpl
#exp_template = chi_sqr_train_full_dla_tmpl


# experiment for chi-square, train and visualize for const ratio, C100
#grid_search_dict = cross_entropy_sgd_train_full_and_vis_grid_search
#exp_template = chi_sqr_sgd_train_full_const_ratio_tmpl

# experiment for chi-square, train and visualize for const ratio, C10
#grid_search_dict = cross_entropy_sgd_train_full_and_vis_grid_search
#exp_template = chi_sqr_sgd_train_full_const_ratio_c10_tmpl


###########

# experiment for cross entropy, train and visualize like deep inversion C100
#grid_search_dict = cross_entropy_sgd_train_full_and_vis_grid_search
#exp_template = cross_entropy_sgd_train_full_and_vis_tmpl

# experiment for cross entropy, train and visualize like deep inversion C10
#grid_search_dict = cross_entropy_sgd_train_full_and_vis_grid_search
#exp_template = cross_entropy_sgd_train_full_and_vis_c10_tmpl

###########

# experiment for chi-square, train and visualize like deep inversion C100
#grid_search_dict = chi_sqr_sgd_train_full_and_vis_grid_search
#exp_template = chi_sqr_sgd_train_full_and_vis_tmpl

# experiment for chi-square, train and visualize like deep inversion C10
#grid_search_dict = chi_sqr_sgd_train_full_and_vis_c10_grid_search
#exp_template = chi_sqr_sgd_train_full_and_vis_c10_tmpl

###########

# experiment for cross-entropy, train and visualize like deep inversion
# in 2 tasks, multitask, C100
#grid_search_dict = cross_entropy_sgd_train_full_and_vis_multitask_grid_search
#exp_template = cross_entropy_sgd_train_full_and_vis_multitask_tmpl

# experiment for cross-entropy, train and visualize like deep inversion
# in 2 tasks, multitask, C10
#grid_search_dict = cross_entropy_sgd_train_full_and_vis_multitask_grid_search
#exp_template = cross_entropy_sgd_train_full_and_vis_multitask_c10_tmpl

###########

# experiment for chi-square, train and visualize like deep inversion
# in 2 tasks, multitask, C100
#grid_search_dict = chi_sqr_sgd_train_full_and_vis_multitask_grid_search
#exp_template = chi_sqr_sgd_train_full_and_vis_multitask_tmpl

# experiment for chi-square, train and visualize like deep inversion
# in 2 tasks, multitask, C10
#grid_search_dict = chi_sqr_sgd_train_full_and_vis_multitask_c10_grid_search
#exp_template = chi_sqr_sgd_train_full_and_vis_multitask_c10_tmpl

main()