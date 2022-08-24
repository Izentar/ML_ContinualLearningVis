from numpy import random
from lucent.optvis import objectives
import math
import numpy as np
from lucent.optvis.objectives import wrap_objective, handle_batch
from torch.utils.data import Subset, random_split, ConcatDataset
import torch

def classic_tasks_split(num_classes, num_tasks):
    # [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    if(num_classes < num_tasks):
        raise Exception(f"Bad number of classes: {num_classes} and tasks: {num_tasks}")
    one_split = num_classes // num_tasks
    ret = [list(range(i * one_split, (i + 1) * one_split)) for i in range(num_tasks)]
    diff = num_classes - one_split * num_tasks
    if(diff == 0):
        return ret
    ret.append(list(range(num_tasks * one_split, num_tasks * one_split + diff)))
    return ret

def decremental_tasks_split(num_classes, num_tasks, jump=2):
    # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9], [6, 7, 8, 9], [8, 9]]
    if(np.ceil(num_classes / jump) < num_tasks):
        raise Exception(f"Bad number of classes: {num_classes}, tasks: {num_tasks}, jump: {jump}")
    return [list(range(num_classes))[i * jump :] for i in range(num_tasks)]

#--------------------------------------------------------------

def classic_select_tasks(tasks, task_index):
    """
        Get only current task set.
    """
    current_split = set(tasks[task_index])
    return current_split

def decremental_select_tasks(tasks, task_index):
    """
        Get difference between previous and current tasks.
    """
    current_split = set(tasks[task_index])
    if(task_index == 0):
        return current_split
    previous_split = set(tasks[task_index - 1])
    return previous_split - current_split

def with_memory_select_tasks(tasks, task_index):
    """
        Get difference between all of the previous tasks and current tasks.
    """
    if(task_index == 0):
        return set(tasks[task_index])
    accumulator = set()
    for idx, t in enumerate(tasks):
        accumulator = accumulator.union(set(t))    
        if idx == task_index:
            break
    current_split = set(tasks[task_index])
    return accumulator - current_split

def with_fading_memory_select_tasks(tasks, task_index, fading_scale, random_distr_f=None):
    """
        Get difference between all of the previous tasks and current tasks.
        fading_scale: [0.0, 1.0], uses geometric sequence to calculate probability threshold of
            including tasks. The 0.0 means all tasks have the same probability drawn from random_distr_f and 
            1.0 means all of the past tasks will not be taken into account except current task that will always be included.
            The 0.25 means that the probability of tasks 0, 1, 2 (task_index = 2) will be
            0.0625, 0.25, 1.0
        random_distr_f: function that returns random variable from some selected distribution.
            Function signature looks like fun(lower_boundary, upper_boundary).
            Use lambda expression to easily pass function.
            Default np.random.uniform.
    """
    if(task_index == 0):
        return set(tasks[task_index])
    accumulator = set()
    if(random_distr_f is None):
        random_distr_f = np.random.uniform
    for idx, t in enumerate(tasks):
        if idx == task_index:
            break
        rand_val = random_distr_f(0.0, 1.0)
        fading = math.pow(fading_scale, (task_index - idx))
        prob_threshold = fading if idx != task_index else 0.0 # special case
        if(rand_val <= prob_threshold):
            accumulator = accumulator.union(set(t))
    current_split = set(tasks[task_index])
    return current_split - accumulator

#--------------------------------------------------------------

#class TaskProcessByLossObj():
#    def __init__(self, loss_obj, distribution_tasks_processing_f):
#        self.loss_obj = loss_obj
#        self.distribution_tasks_processing_f = distribution_tasks_processing_f
#
#    def __call__(self, tasks):
#        return self.distribution_tasks_processing_f(tasks, self.loss_obj)

def default_tasks_processing(tasks, *args, **kwargs):
    return tasks

def normall_dist_tasks_processing(tasks, mean, std):
    """
        Return a set of points taken from the normal distribution.
        Mean and std already have the shape of <class <mean / std of the points in this class>> so
        the tasks variable is excessive. It always calculate for the all tasks, because it is faster 
        than using python loop.
        Function returns an array <class <calculated point for this class>>.
        If you want to have a constant value of std, then use lambda expression to set an std to
        torch.tensor([std_value]). It will repeat it to the size of the mean. 
    """
    size = list(std.size())
    if len(size) == 0 or size[0] == 1:
        std = std.repeat(mean.size()[0])
    tmp = torch.normal(mean=mean, std=std)
    if isinstance(tasks, list):
        return torch.index_select(tmp, dim=0, index=torch.tensor(tasks))
    return tmp[tasks]

#--------------------------------------------------------------

@wrap_objective()
def multidim_objective_channel(layer, batch=None):
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[:, ].mean() #TODO - czy to jest prawidłowe, gdy n_channel nie jest używane? Chcemy wszystkie "punkty"
    return inner


def SAE_standalone_multidim_dream_objective_f(target, model, source_dataset_obj):
    return multidim_objective_channel(model.get_objective_target()) - objectives.diversity(
        "model_conv2"
    )

def SAE_multidim_dream_objective_f(target, model, source_dataset_obj):
    return multidim_objective_channel(model.get_objective_target()) - objectives.diversity(
        "model_model_conv2"
    )

def SAE_dream_objective_f(target, model, source_dataset_obj):
    # be careful for recursion by calling methods from source_dataset_obj
    # specify layers names from the model - <top_var_name>_<inner_layer_name>
    # and apply the objective on this layer. Used only for dreams.
    # channel - diversity penalty
    return objectives.channel(model.get_objective_target(), target) - objectives.diversity(
        "model_model_conv2"
    )

def default_dream_objective_f(target, model, source_dataset_obj):
    return objectives.channel(model.get_objective_target(), target)

def test(target, model, source_dataset_obj):
    return objectives.channel(model.get_objective_target(), target) - objectives.diversity(
        "vgg_features_10"
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


#--------------------------------------------------

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

def select_class_indices_tensor(cl, target):
    cl_indices = torch.isin(target, cl)
    cl_indices_list = torch.where(cl_indices)[0]
    return cl_indices_list

def select_class_indices_numpy(cl, target):
    cl_indices = np.isin(np.array(target), cl)
    cl_indices_list = np.where(cl_indices)[0]
    return cl_indices_list