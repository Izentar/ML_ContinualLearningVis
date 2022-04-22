
# pass modules names like torch.autograd.variable
# do not import classes
# you can write multiple modules in command line

from torch.autograd import variable

if __name__ == "__main__":
    import sys
    import importlib
    import pprint

    for idx, arg in enumerate(sys.argv[1:]):
        lib = importlib.import_module(arg)

        print(lib.__file__) if hasattr(lib, "__file__") else print("No attribute: __file__")
        print(lib.__name__) if hasattr(lib, "__name__") else print("No attribute: __name__")
        print(lib.__package__) if hasattr(lib, "__package__") else print("No attribute: __package__")
        print(lib.__version__) if hasattr(lib, "__version__") else print("No attribute: __version__")
        d = dict(lib.__dict__)
        print(pprint.pformat(d))

    # redirect to file ( .py > txt)