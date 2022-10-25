import torch


def task_processing_default(target, *args, **kwargs):
    return target

def island_task_processing_decode(target, model, *args, **kwargs):
    return model.decode(torch.tensor([target], dtype=torch.int32)).float()

def island_task_processing_sample_normal_vectors(target_point, std_vector, *args, **kwargs):
    return torch.normal(target_point, std_vector)

def task_processing_normal_distr_mean_std(target, mean, std, *args, **kwargs):
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
        std = std.repeat(mean.size()[0]) / 2 # TODO div 2 not needed?
    tmp = torch.normal(mean=mean, std=std)
    if isinstance(target, list):
        return torch.index_select(tmp, dim=0, index=torch.tensor(target))
    return tmp[target]

def island_task_processing_sample_surroundings_std_mean_vectors(mean_vector, std_vector, *args, **kwargs):
    assert torch.all(std_vector >= 0.0), f"Bad value mean/std \n{mean_vector} \n{std_vector}"
    point = torch.normal(mean=mean_vector, std=std_vector)
    return point

def island_task_processing_sample_surroundings_std_mean(target, model, *args, **kwargs):
    std, mean = model.get_buffer().std_mean_target(target)
    assert torch.all(std >= 0.0), f"Bad value mean/std \n{mean} \n{std} \n{target}"
    return island_task_processing_sample_surroundings_std_mean_vectors(mean_vector=mean, std_vector=std)

def island_task_processing_sample_from_cov(target, model, *args, **kwargs):
    sample = model.loss_f.sample(target)
    return sample

def island_task_processing_get_last_point(target, model, *args, **kwargs):
    last_point = model.get_buffer().front(target)
    return last_point

def island_task_processing_get_mean(target, model, *args, **kwargs):
    classes_mean = model.get_buffer().mean(target)
    return classes_mean[target]