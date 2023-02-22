from latent_dreams import logic
from my_parser import arg_parser

if __name__ == "__main__":
    args = arg_parser()
    logic(args, True)