import torch

def target_processing_latent_binary_classification(target, model, *args, **kwargs):
    out = torch.zeros(model.get_objective_layer_output_shape(), dtype=torch.float32)
    out[target] = 1.
    return out

def target_processing_default(target, *args, **kwargs):
    return torch.tensor(target, dtype=torch.float32)

def target_processing_latent_decode(target, model, *args, **kwargs):
    return model.decode(torch.tensor([target], dtype=torch.int32)).float()

def target_processing_latent_sample_normal_std(mean, std, *args, **kwargs):
    return torch.normal(mean, std)

def target_processing_latent_sample_normal_mean_std_full_targets(target, mean, std, *args, **kwargs):
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

def target_processing_latent_sample_normal_mean_std_vectors(mean, std, *args, **kwargs):
    assert torch.all(std >= 0.0), f"Bad value mean/std \n{mean} \n{std}"
    point = torch.normal(mean=mean, std=std)
    return point

def target_processing_latent_sample_normal_buffer(target, model, *args, **kwargs):
    std, mean = model.get_buffer().std_mean_target(target)
    assert torch.all(std >= 0.0), f"Bad value mean/std \n{mean} \n{std} \n{target}"
    return target_processing_latent_sample_normal_mean_std_vectors(mean=mean, std=std)

def target_processing_latent_sample_multivariate(target, model, *args, **kwargs):
    sample = model.loss_f.sample(target)
    return sample

def target_processing_latent_buffer_last_point(target, model, *args, **kwargs):
    last_point = model.get_buffer().front(target)
    return last_point

def target_processing_latent_mean(target, model, *args, **kwargs):
    classes_mean = model.get_buffer().mean(target)
    return classes_mean[target]


class TargetProcessingManager():
    GET_TARGET_PROCESSING = {
        'TARGET-DEFAULT': target_processing_default,
        'TARGET-LATENT-DECODE': target_processing_latent_decode,
        'TARGET-LATENT-SAMPLE-NORMAL-STD': target_processing_latent_sample_normal_std,
        'TARGET-LATENT-SAMPLE-NORMAL-MEAN-STD-FULL-TARGETS': target_processing_latent_sample_normal_mean_std_full_targets,
        'TARGET-LATENT-SAMPLE-NORMAL-MEAN-STD-VECTORS': target_processing_latent_sample_normal_mean_std_vectors,
        'TARGET-LATENT-SAMPLE-NORMAL-BUFFER': target_processing_latent_sample_normal_buffer,
        'TARGET-LATENT-SAMPLE-MULTIVARIATE': target_processing_latent_sample_multivariate,
        'TARGET-LATENT-BUFFER-LAST-POINT': target_processing_latent_buffer_last_point,
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