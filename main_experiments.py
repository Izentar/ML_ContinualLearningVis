from latent_dreams import logic
import my_parser

if __name__ == "__main__":
    parser = my_parser.main_arg_parser()
    parser = my_parser.layer_statistics_arg_parser(parser)
    parser = my_parser.dream_parameters_arg_parser(parser)
    #parser = my_parser.dual_optim_arg_parser(parser)
    parser = my_parser.model_statistics_optim_arg_parser(parser)
    args = my_parser.parse_args(parser)

    logic(args, True)