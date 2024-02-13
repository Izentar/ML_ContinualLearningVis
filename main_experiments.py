from latent_dreams import logic
import my_parser
import gc
import torch
import shlex
import traceback
from datetime import datetime

def main(fast_dev_run=True):
    """
        Change fast_dev_run flag to run full experiments.
    """
    if __name__ == "__main__":
        REPEAT = 3
        
        parser = my_parser.main_arg_parser()
        parser = my_parser.layer_statistics_arg_parser(parser)
        parser = my_parser.dream_parameters_arg_parser(parser)
        parser = my_parser.model_statistics_optim_arg_parser(parser)

        time = datetime.today().strftime('%Y-%m-%d=%H-%M-%S')

        print(f"Experiments to be run:")
        for v in experiments.values():
            print(f"\t* {v}")

        print()
        for k, v in experiments.items():
            for idx in range(1, REPEAT + 1):
                print(f"Running experiment: {k}; repeat {idx}/{REPEAT}")

                v = v.replace('\n', ' ')
                if(fast_dev_run):
                    v += ' -f'
                args_exp = my_parser.parse_args(parser, shlex.split(v))
                try:
                    logic(args_exp, True, project_name=f"exp_{time}", run_name=k)
                except Exception:
                    print("Experiment exception occurred")
                    print(traceback.format_exc())

                print(f"End of experiment: {k}; repeat {idx}/{REPEAT}")
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
--loop.save.root model_save/test --loop.save.model \
"""

crossentropy_default_c100_sgd = """
-d c100 --model.num_classes 100 --loop.schedule 200 \
--config.framework_type crossentropy-default --model.type \
custom-resnet34 --loop.num_loops 1 --loop.train_at 0 \
--model.optim.type sgd --model.optim.kwargs.lr 0.1 \
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1 \
--model.sched.kwargs.milestones 100 150 --datamodule.num_workers 3 \
--loop.save.root model_save/test --loop.save.model \
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
--model.loss.chi.scale 15 --model.loss.chi.ratio_gamma 2 \
--model.loss.chi.ratio_milestones 5 20 40 60 \
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
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 5 \
--model.loss.chi.scale 40 --model.loss.chi.ratio_gamma 2 \
--model.loss.chi.ratio_milestones 5 20 40 60 \
"""

experiments = {
    "crossentropy_default_c10_sgd": crossentropy_default_c10_sgd,
    "crossentropy_default_c100_sgd": crossentropy_default_c10_sgd,
    "chi_sqr_c10_sgd": chi_sqr_c10_sgd,
    "chi_sqr_c100_sgd": chi_sqr_c100_sgd,
}




main()