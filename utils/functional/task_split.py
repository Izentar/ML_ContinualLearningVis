import numpy as np


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
