from latent_dreams import logic
from my_parser import arg_parser, log_to_wandb

if __name__ == "__main__":
    args = arg_parser()
    log_to_wandb(args)
    logic(args)