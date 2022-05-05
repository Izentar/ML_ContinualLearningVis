from numpy import random
from lucent.optvis import objectives
import math
import numpy
from lucent.optvis.objectives import wrap_objective, handle_batch

def classic_tasks_split(num_classes, num_tasks):
    # [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    # for multidim tasks this would look like (task_id, mean[vector], var[vector])
    # where the vector represents a point in multidimensional plane
    # [
    #   [
    #       (0, (0.2, 2.4, 8.9), (3.5, 5.7, 9.4)), 
    #       (1, (0.3, 7.6, 1.5), (4.5, 8.5, 7.3))
    #   ], 
    #   [
    #       (2, (4, 5.1, 3.7), (0.5, 8.1, 6.1)), 
    #       (3, (2.1, 8.2, 1.6), (2.4, 9.1, 2.2))
    #   ]
    # ]
    one_split = num_classes // num_tasks
    return [list(range(i * one_split, (i + 1) * one_split)) for i in range(num_tasks)]


def decremental_tasks_split(num_classes, num_tasks):
    # [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [2, 3, 4, 5, 6, 7, 8, 9], [4, 5, 6, 7, 8, 9], [6, 7, 8, 9], [8, 9]]
    return [list(range(num_classes))[i * 2 :] for i in range(num_tasks)]

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
    accumulator = set()
    for idx, t in enumerate(tasks):
        accumulator += set(t)
        if idx == task_index:
            break
    current_split = set(tasks[task_index])
    return accumulator - current_split

def with_fading_memory_select_tasks(tasks, task_index, fading_scale, random_distr_f):
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
    """
    accumulator = set()
    for idx, t in enumerate(tasks):
        rand_val = random_distr_f(0.0, 1.0)
        fading = math.pow(fading_scale, (task_index - (idx + 1)))
        prob_threshold = fading if idx != task_index else 0.0 # special case
        
        if( rand_val >= prob_threshold):
            accumulator += set(t)
        if idx == task_index:
            break
    current_split = set(tasks[task_index])
    return accumulator - current_split

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

def normall_dist_tasks_processing(tasks, get_mean_var_f):
    """
        Return a set of points taken from the normal distribution.
        The mean and variance will be taken from get_mean_var function that
        have signature of f(task_id).
        Use lambda to pass this function into invoker.
    """
    #if not hasattr(source_obj, "get_mean_var"):
    #    raise Exception(f"Passed object {type(source_obj)} must implement the get_mean_var(task) "
    #        "functions from loss_function.point_scope.AbstractNormalDistr")
    out = []
    for t in tasks:
        processed_task_mean, processed_task_var = get_mean_var_f(t)
        point = []
        for m, v in zip(processed_task_mean, processed_task_var):
            value = numpy.random.normal(loc=m, scale=v)
            point.append(value)
        out.append(point)
    return out

#--------------------------------------------------------------

@wrap_objective()
def multidim_objective_channel(layer, n_channel, batch=None):
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[:, ].mean()
    return inner


def SAE_multidim_dream_objective_f(target, model, source_dataset_obj):
    return multidim_objective_channel(model.get_objective_target(), target) - objectives.diversity(
        "sae_conv2"
    )

def SAE_dream_objective_f(target, model, source_dataset_obj):
    # be carrefour for recursion by calling methods from source_dataset_obj
    # specify layers names from the model - <top_var_name>_<inner_layer_name>
    # and apply the objective on this layer. Used only for dreams.
    # channel - diversity penalty
    return objectives.channel(model.get_objective_target(), target) - objectives.diversity(
        "sae_conv2"
    )

def default_dream_objective_f(target, model, source_dataset_obj):
    return objectives.channel(model.get_objective_target(), target)

def test(target, model, source_dataset_obj):
    return objectives.channel(model.get_objective_target(), target) - objectives.diversity(
        "vgg_features_10"
    )

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)