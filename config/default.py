from torchvision.datasets import CIFAR10, CIFAR100
from robustness.datasets import (
    CIFAR10 as CIFAR10_robust,
    CIFAR100 as CIFAR100_robust
)

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