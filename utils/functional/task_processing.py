import torch


def default_tasks_processing(target, *args, **kwargs):
    return target

def normall_dist_tasks_processing(target, mean, std, *args, **kwargs):
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
    if isinstance(target, list):
        return torch.index_select(tmp, dim=0, index=torch.tensor(target))
    return tmp[target]

def normall_dist_tasks_processing_vector(mean_vector, std_vector, *args, **kwargs):
    assert torch.all(std_vector >= 0.0), f"Bad value mean/std \n{mean_vector} \n{std_vector}"
    point = torch.normal(mean=mean_vector, std=std_vector)
    return point

def island_tasks_processing(target, model, *args, **kwargs):
    classes_std_mean = model.get_buffer().std_mean()
    std, mean = classes_std_mean[target]
    assert torch.all(std >= 0.0), f"Bad value mean/std \n{mean} \n{std} \n{target}"
    return normall_dist_tasks_processing_vector(mean_vector=mean, std_vector=std)

def island_last_point_tasks_processing(target, model, *args, **kwargs):
    last_point = model.get_buffer().front(target)
    return last_point

def island_mean_tasks_processing(target, model, *args, **kwargs):
    classes_mean = model.get_buffer().mean()
    return classes_mean[target]