import numpy as np
import math

def select_task_classic(tasks, task_index):
    """
        Get only current task set.
    """
    current_split = set(tasks[task_index])
    return current_split

def select_task_decremental(tasks, task_index):
    """
        Get difference between previous and current tasks.
    """
    if(len(tasks) <= task_index):
        return set(tasks[len(tasks) - 1])
    current_split = set(tasks[task_index])
    if(task_index == 0):
        return current_split
    previous_split = set(tasks[task_index - 1])
    return previous_split - current_split

def select_task_with_memory(tasks, task_index):
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

def select_task_with_fading_memory(tasks, task_index, fading_scale, random_distr_f=None):
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
