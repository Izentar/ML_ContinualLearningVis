from torchvision.datasets import CIFAR10, CIFAR100
from robustness.datasets import (
    CIFAR10 as CIFAR10_robust,
    CIFAR100 as CIFAR100_robust
)
import itertools
from pathlib import Path

fast_dev_run_config = {
    "num_tasks": 1,
}

optim_Adam_config = {
    "lr": 1e-3
}

CIFAR10_labels = {
        0 : 'airplane',
        1 : 'automobile',
        2 : 'bird',
        3 : 'cat',
        4 : 'deer',
        5 : 'dog',
        6 : 'frog',
        7 : 'horse',
        8 : 'ship',
        9 : 'truck',
    }

CIFAR100_labels = {
    0: "apple",
    1: "aquarium_fish",
    2: "baby",
    3: "bear",
    4: "beaver",
    5: "bed",
    6: "bee",
    7: "beetle",
    8: "bicycle",
    9: "bottle",
    10: "bowl",
    11: "boy",
    12: "bridge",
    13: "bus",
    14: "butterfly",
    15: "camel",
    16: "can",
    17: "castle",
    18: "caterpillar",
    19: "cattle",
    20: "chair",
    21: "chimpanzee",
    22: "clock",
    23: "cloud",
    24: "cockroach",
    25: "couch",
    26: "crab",
    27: "crocodile",
    28: "cup",
    29: "dinosaur",
    30: "dolphin",
    31: "elephant",
    32: "flatfish",
    33: "forest",
    34: "fox",
    35: "girl",
    36: "hamster",
    37: "house",
    38: "kangaroo",
    39: "keyboard",
    40: "lamp",
    41: "lawn_mower",
    42: "leopard",
    43: "lion",
    44: "lizard",
    45: "lobster",
    46: "man",
    47: "maple_tree",
    48: "motorcycle",
    49: "mountain",
    50: "mouse",
    51: "mushroom",
    52: "oak_tree",
    53: "orange",
    54: "orchid",
    55: "otter",
    56: "palm_tree",
    57: "pear",
    58: "pickup_truck",
    59: "pine_tree",
    60: "plain",
    61: "plate",
    62: "poppy",
    63: "porcupine",
    64: "possum",
    65: "rabbit",
    66: "raccoon",
    67: "ray",
    68: "road",
    69: "rocket",
    70: "rose",
    71: "sea",
    72: "seal",
    73: "shark",
    74: "shrew",
    75: "skunk",
    76: "skyscraper",
    77: "snail",
    78: "snake",
    79: "spider",
    80: "squirrel",
    81: "streetcar",
    82: "sunflower",
    83: "sweet_pepper",
    84: "table",
    85: "tank",
    86: "telephone",
    87: "television",
    88: "tiger",
    89: "tractor",
    90: "train",
    91: "trout",
    92: "tulip",
    93: "turtle",
    94: "wardrobe",
    95: "whale",
    96: "willow_tree",
    97: "wolf",
    98: "woman",
    99: "worm",
}

datasets = {
    "TORCH_CIFAR100": [CIFAR100, CIFAR100_labels],
    "TORCH_CIFAR10": [CIFAR10, CIFAR10_labels],
    "CIFAR100": [CIFAR100, CIFAR100_labels],
    "CIFAR10": [CIFAR10, CIFAR10_labels],
    "C100": [CIFAR100, CIFAR100_labels],
    "C10": [CIFAR10, CIFAR10_labels],
    "ROBUST_CIFAR100": [CIFAR100_robust, CIFAR100_labels],
    "ROBUST_CIFAR10": [CIFAR10_robust, CIFAR10_labels],
    "RC100": [CIFAR100_robust, CIFAR100_labels],
    "RC10": [CIFAR10_robust, CIFAR10_labels]
}

datasets_map = {
    "TORCH_CIFAR100": [CIFAR100, CIFAR100_robust],
    "TORCH_CIFAR10": [CIFAR10, CIFAR10_robust],
    "CIFAR100": [CIFAR100, CIFAR100_robust],
    "CIFAR10": [CIFAR10, CIFAR10_robust],
    "C100": [CIFAR100, CIFAR100_robust],
    "C10": [CIFAR10, CIFAR10_robust],
    "ROBUST_CIFAR100": [CIFAR100, CIFAR100_robust],
    "ROBUST_CIFAR10": [CIFAR10, CIFAR10_robust],
    "RC100": [CIFAR100, CIFAR100_robust],
    "RC10": [CIFAR10, CIFAR10_robust]
}

default_export_path = './model_save/'
model_to_save_file_type = 770
robust_data_path = "./data"
tmp_stat_folder = 'tmp/stats/'


colors_list = ('r', 'g', 'b', 'c', 'k', 'm', 'y', 'indianred', 'salmon', 'darkkhaki', 'violet')
markers_list = ('>', '+', '.', 'o', '*')
markers = itertools.cycle(markers_list)
colors = itertools.cycle(colors_list)

Path(tmp_stat_folder).mkdir(exist_ok=True, parents=True)