from latent_dreams import logic
import my_parser
import gc
import torch
import shlex
import traceback
from datetime import datetime
from argparse import ArgumentParser, Namespace

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

        time = datetime.today().strftime('%Y-%m-%d=%H-%M-%S')

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
                        project_name = f"exp_{time}"
                    
                    logic(args_exp, True, project_name=project_name, run_name=k)
                except Exception:
                    print("Experiment exception occurred")
                    print(traceback.format_exc())

                print(f"End of experiment: {k}; repeat {idx}/{args.repeat}")
                print("Clearing gpu cache and invoking garbage collector")

                torch.cuda.empty_cache()
                gc.collect()
                print("Done")
                

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

chi_sqr_c100_sgd_search_1 = """
-d c100 --model.num_classes 100 --loop.schedule 260 \
--config.framework_type latent-multitarget --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --model.latent.size 20 --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 10 \
--model.loss.chi.scale 100 --model.loss.chi.ratio_gamma 2 --datamodule.batch_size 320 \
--model.loss.chi.ratio_milestones 5 20 40 60 --config.seed 2024 \
"""

chi_sqr_c100_sgd_search_2 = """
-d c100 --model.num_classes 100 --loop.schedule 260 \
--config.framework_type latent-multitarget --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --model.latent.size 10 --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 10 \
--model.loss.chi.scale 100 --model.loss.chi.ratio_gamma 2 --datamodule.batch_size 320 \
--model.loss.chi.ratio_milestones 5 20 40 60 --config.seed 2024 \
"""


chi_sqr_c100_sgd_search_3 = """
-d c100 --model.num_classes 100 --loop.schedule 260 \
--config.framework_type latent-multitarget --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --model.latent.size 3 --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 10 \
--model.loss.chi.scale 100 --model.loss.chi.ratio_gamma 2 --datamodule.batch_size 320 \
--model.loss.chi.ratio_milestones 5 20 40 60 --config.seed 2024 \
"""


chi_sqr_c100_sgd_search_4 = """
-d c100 --model.num_classes 100 --loop.schedule 260 \
--config.framework_type latent-multitarget --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --model.latent.size 30 --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 10 \
--model.loss.chi.scale 100 --model.loss.chi.ratio_gamma 2 --datamodule.batch_size 256 \
--model.loss.chi.ratio_milestones 5 20 40 60 --config.seed 2024 \
"""


chi_sqr_c100_sgd_search_5 = """
-d c100 --model.num_classes 100 --loop.schedule 260 \
--config.framework_type latent-multitarget --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --model.latent.size 30 --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 10 \
--model.loss.chi.scale 100 --model.loss.chi.ratio_gamma 2 --datamodule.batch_size 200 \
--model.loss.chi.ratio_milestones 5 20 40 60 --config.seed 2024 \
"""

chi_sqr_c100_sgd_search_6 = """
-d c100 --model.num_classes 100 --loop.schedule 260 \
--config.framework_type latent-multitarget --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --model.latent.size 30 --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 10 \
--model.loss.chi.scale 100 --model.loss.chi.ratio_gamma 2 --datamodule.batch_size 150 \
--model.loss.chi.ratio_milestones 5 20 40 60 --config.seed 2024 \
"""

chi_sqr_c100_sgd_search_7 = """
-d c100 --model.num_classes 100 --loop.schedule 260 \
--config.framework_type latent-multitarget --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model --loop.load.root \
model_save/test --model.latent.size 30 --stat.collect_stats.enable \
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 10 \
--model.loss.chi.scale 100 --model.loss.chi.ratio_gamma 2 --datamodule.batch_size 100 \
--model.loss.chi.ratio_milestones 5 20 40 60 --config.seed 2024 \
"""



experiments = {
    "crossentropy_default_c10_sgd": crossentropy_default_c10_sgd,
    "crossentropy_default_c100_sgd": crossentropy_default_c10_sgd,
    "chi_sqr_c10_sgd": chi_sqr_c10_sgd,
    "chi_sqr_c100_sgd": chi_sqr_c100_sgd,
    "chi_sqr_c100_sgd_search_1": chi_sqr_c100_sgd_search_1,
    "chi_sqr_c100_sgd_search_2": chi_sqr_c100_sgd_search_2,
    "chi_sqr_c100_sgd_search_3": chi_sqr_c100_sgd_search_3,
    "chi_sqr_c100_sgd_search_4": chi_sqr_c100_sgd_search_4,
    "chi_sqr_c100_sgd_search_5": chi_sqr_c100_sgd_search_5,
    "chi_sqr_c100_sgd_search_6": chi_sqr_c100_sgd_search_6,
    "chi_sqr_c100_sgd_search_7": chi_sqr_c100_sgd_search_7,
}




main()