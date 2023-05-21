import torch
import numpy as np

"""
    Called after select_task.py
    Strategy for processing given target.
    Can return the same target or can sample it from chiloss, giving a vector from latent space.
"""

def target_processing_latent_binary_classification(target, model):
    out = torch.zeros(model.get_objective_layer_output_shape(), dtype=torch.float32)
    out[target] = 1.
    return out

def target_processing_default(target, model):
    # if autograd error 'Found dtype Int but expected Float', check other config framework parameters, good example "cl-sae-crossentropy"
    return torch.tensor(target, dtype=torch.int64)

def target_processing_latent_decode(target, model):
    return model.decode(torch.tensor([target], dtype=torch.int32)).float()

def target_processing_latent_sample_normal_std(target, model):
    std, mean = model.loss_f.cloud_data.std_mean_target(target)
    assert torch.all(std >= 0.0), f"Bad value mean/std \n{mean} \n{std} \n{target}"
    return torch.normal(mean, std)

def target_processing_latent_sample_normal_std_multitarget(target, model):
    sum_mean = []
    sum_std = []
    for t in target:
        std, mean = model.loss_f.cloud_data.std_mean_target(t)
        assert torch.all(std >= 0.0), f"Bad value mean/std \n{mean} \n{std} \n{t}"
        sum_mean.append(mean)
        sum_std.append(std)
    return torch.normal(torch.stack(sum_mean), torch.stack(sum_std))

def target_processing_latent_sample_normal_std_multitarget_func(target, model):
    def inner():
        sum_mean = []
        sum_std = []
        for t in target:
            std, mean = model.loss_f.cloud_data.std_mean_target(t)
            assert torch.all(std >= 0.0), f"Bad value mean/std \n{mean} \n{std} \n{t}"
            sum_mean.append(mean)
            sum_std.append(std)
        return torch.normal(torch.stack(sum_mean), torch.stack(sum_std))
    return inner

def target_processing_latent_sample_normal_mean_std_full_targets(target, model):
    """
        Return a set of points taken from the normal distribution.
        Mean and std already have the shape of <class <mean / std of the points in this class>> so
        the tasks variable is excessive. It always calculate for the all tasks, because it is faster 
        than using python loop.
        Function returns an array <class <calculated point for this class>>.
        If you want to have a constant value of std, then use lambda expression to set an std to
        torch.tensor([std_value]). It will repeat it to the size of the mean. 
    """
    std, mean = model.loss_f.cloud_data.std_mean()
    size = list(std.size())
    if len(size) == 0 or size[0] == 1:
        std = std.repeat(mean.size()[0]) / 2 # TODO div 2 not needed?
    tmp = torch.normal(mean=mean, std=std)
    if isinstance(target, list):
        return torch.index_select(tmp, dim=0, index=torch.tensor(target))
    return tmp[target]

def target_processing_latent_sample_multivariate(target, model):
    """
        Takes a bit to compute
    """
    sample = model.loss_f.sample(target)
    return sample

def target_processing_latent_buffer_last_point(target, model):
    last_point = model.loss_f.cloud_data.front(target)
    return last_point

def target_processing_latent_buffer_random_index(target, model):
    data =  model.loss_f.cloud_data
    size = len(data)
    rand_idx = np.random.randint(0, size)
    last_point = data.get(target, rand_idx)
    return last_point

def target_processing_latent_buffer_last_point_multitarget(target, model):
    tmp = [target_processing_latent_buffer_last_point(t, model) for t in target]
    return torch.stack(tmp)
        
def target_processing_latent_buffer_random_index_multitarget(target, model):
    tmp = [target_processing_latent_buffer_random_index(target, model) for t in target]
    return torch.stack(tmp)

def target_processing_latent_mean(target, model):
    classes_mean = model.loss_f.cloud_data.mean(target)
    return classes_mean[target]


class TargetProcessingManager():
    GET_TARGET_PROCESSING = {
        'TARGET-CLASSIC': target_processing_default,
        'TARGET-LATENT-DECODE': target_processing_latent_decode,
        'TARGET-LATENT-SAMPLE-NORMAL-STD': target_processing_latent_sample_normal_std,
        'TARGET-LATENT-SAMPLE-NORMAL-STD-MULTITARGET': target_processing_latent_sample_normal_std_multitarget,
        'TARGET-LATENT-SAMPLE-NORMAL-STD-MULTITARGET-FUNC': target_processing_latent_sample_normal_std_multitarget_func,
        'TARGET-LATENT-SAMPLE-NORMAL-MEAN-STD-FULL-TARGETS': target_processing_latent_sample_normal_mean_std_full_targets,
        'TARGET-LATENT-SAMPLE-MULTIVARIATE': target_processing_latent_sample_multivariate,
        'TARGET-LATENT-BUFFER-LAST-POINT': target_processing_latent_buffer_last_point,
        'TARGET-LATENT-BUFFER-LAST-POINT-MULTITARGET': target_processing_latent_buffer_last_point_multitarget,
        'TARGET-LATENT-BUFFER-RAND-INDEX': target_processing_latent_buffer_random_index,
        'TARGET-LATENT-BUFFER-RAND-INDEX-MULTITARGET': target_processing_latent_buffer_random_index_multitarget,
        'TARGET-LATENT-MEAN': target_processing_latent_mean,
        'TARGET-LATENT-BINARY-CLASSIFICATION': target_processing_latent_binary_classification
    }

    def __init__(self, dtype: str) -> None:
        dtype = dtype.upper()
        self.target_processing = TargetProcessingManager.GET_TARGET_PROCESSING[dtype]
        self.target_processing_name = dtype

    def __call__(self, *args, **kwargs):
        return self.target_processing(*args, **kwargs)

    def get_name(self) -> str:
        return self.target_processing_name

    def is_latent(self) -> bool:
        return '_latent' in self.target_processing.__name__