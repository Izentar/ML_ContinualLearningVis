from torchvision.datasets import CIFAR10, CIFAR100
from robustness.datasets import (
    CIFAR10 as CIFAR10_robust,
    CIFAR100 as CIFAR100_robust
)
import itertools

fast_dev_run_config = {
    "num_tasks": 1,
}

optim_Adam_config = {
    "lr": 1e-3
}

datasets = {
    "TORCH_CIFAR100": CIFAR100,
    "TORCH_CIFAR10": CIFAR10,
    "CIFAR100": CIFAR100,
    "CIFAR10": CIFAR10,
    "C100": CIFAR100,
    "C10": CIFAR10,
    "ROBUST_CIFAR100": CIFAR100_robust,
    "ROBUST_CIFAR10": CIFAR10_robust,
    "RC100": CIFAR100_robust,
    "RC10": CIFAR10_robust
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


colors_list = ('r', 'g', 'b', 'c', 'k', 'm', 'y', 'indianred', 'salmon', 'darkkhaki', 'violet')
markers_list = ('>', '+', '.', 'o', '*')
markers = itertools.cycle(markers_list)
colors = itertools.cycle(colors_list)