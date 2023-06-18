import numpy as np
from torch.utils.data import Subset, random_split, ConcatDataset
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def split_by_class_dataset(dataset, all_classes):
    # TODO not finished
    split_dataset = []
    for cl in all_classes:
        cl_indices = np.isin(np.array(dataset.targets), [cl])
        cl_indices_list = np.where(cl_indices)[0]
        new_set = Subset(dataset, cl_indices_list)
        split_dataset.append(new_set)
    return split_dataset

def pair_sets(split_dataset, main_task_part, mode='strict'):
    # TODO not finished
    """
        mode: 
            strict - in the remaining (1 - main_task_part) dataset, the main class will not be present.
            loose - in the remaining (1 - main_task_part) dataset, the main class can be still present.
            In any mode, the data sample is present only once.
    """
    split_sets_main = []
    all_except_one_sets = []
    return_sets = []
    other_sets = []
    tasks_numb = len(split_dataset)
    for task_dataset in split_dataset:
        size = len(task_dataset)
        main_size = size * main_task_part
        other_size = size - main_size
        main_task_dataset, other_task_dataset = random_split(task_dataset, [main_size, other_size],
            generator=torch.Generator())
        split_sets_main.append(main_task_dataset)
        other_sets.append(other_task_dataset)

    numb_of_sets = len(split_sets_main)

    if mode == 'strict':
        for idx,(_, _) in enumerate(other_sets):
            tmp = other_sets[:idx] + other_sets[idx+1:]
            all_except_one_sets.append(tmp)
    elif mode == 'loose':
        remaining_dataset = ConcatDataset(other_sets)
        split_by = len(remaining_dataset) / tasks_numb
        all_except_one_sets.extend(random_split(remaining_dataset, [split_by] * tasks_numb, 
            generator=torch.Generator()))

    for main_task_dataset, all_except_one_dataset in zip(split_sets_main, all_except_one_sets):
        return_sets.append(
            ConcatDataset([main_task_dataset, all_except_one_dataset])
        )

    return return_sets

def get_target_from_dataset(dataset, toTensor=False) -> list:
    if isinstance(dataset, torch.utils.data.Subset):
        target_subset = np.take(dataset.dataset.targets, dataset.indices)
        if toTensor:
            return torch.tensor(target_subset)
        return target_subset.tolist()
    if toTensor:
        return torch.tensor(dataset.targets)
    return dataset.targets

def select_class_indices_tensor(cl, target: torch.Tensor):
    cl_indices = torch.isin(target, cl)
    cl_indices_list = torch.where(cl_indices)[0]
    return cl_indices_list

def select_class_indices_numpy(cl, target):
    cl_indices = np.isin(np.array(target), cl)
    cl_indices_list = np.where(cl_indices)[0]
    return cl_indices_list