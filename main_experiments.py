from latent_dreams import logic
import my_parser
import gc
import torch
import shlex
import traceback
from datetime import datetime

def main():
    if __name__ == "__main__":
        REPEAT = 1
        
        parser = my_parser.main_arg_parser()
        parser = my_parser.layer_statistics_arg_parser(parser)
        parser = my_parser.dream_parameters_arg_parser(parser)
        parser = my_parser.model_statistics_optim_arg_parser(parser)

        time = datetime.today().strftime('%Y-%m-%d=%H-%M-%S')
        
        for k, v in experiments.items():
            for idx in range(1, REPEAT + 1):
                print(f"Running experiment: {k}; repeat {idx}/{REPEAT}")

                v = v.replace('\n', ' ')
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
                

experiments = {
    "exp_1": """
-d c10 --model.num_classes 10 --loop.schedule 200
--config.framework_type crossentropy-default --model.type
custom-resnet34 --loop.num_loops 1 --loop.train_at 0
--model.optim.type sgd --model.optim.kwargs.lr 0.1
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1
--model.sched.kwargs.milestones 100 150 --datamodule.num_workers 3
--loop.save.root model_save/test --loop.save.model -f
""",

    "exp_2":  """
-d c10 --model.num_classes 10 --loop.schedule 260
--config.framework_type latent-multitarget --model.type
custom-resnet34 --loop.num_loops 1 --loop.train_at 0
--model.optim.type sgd --model.optim.kwargs.lr 0.1
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1
--model.sched.kwargs.milestones 140 180 --datamodule.num_workers 3
--loop.save.root model_save/test --loop.save.model --loop.load.root
model_save/test --model.latent.size 3 --stat.collect_stats.enable
--model.loss.chi.shift_min_distance 0 --model.loss.chi.ratio 10
--model.loss.chi.scale 10 --model.loss.chi.ratio_gamma 2
--model.loss.chi.ratio_milestones 5 20 40 60 -f
""",

"exp_3": """
-d c10 --model.num_classes 10 --loop.schedule 200
--config.framework_type crossentropy-default --model.type
custom-resnet34 --loop.num_loops 1 --loop.train_at 0
--model.optim.type sgd --model.optim.kwargs.lr 0.1
--model.sched.type MULTISTEP-SCHED --model.sched.kwargs.gamma 0.1
--model.sched.kwargs.milestones 100 150 --datamodule.num_workers 3
--loop.save.root model_save/test --loop.save.model -f
""",
}



main()