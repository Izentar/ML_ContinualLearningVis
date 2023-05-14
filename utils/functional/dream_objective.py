import torch
from lucent.optvis.objectives import wrap_objective, handle_batch
from lucent.optvis import objectives
from utils.counter import Counter, CounterBase, CounterKeys, CounterKeysBase
import torch.nn.functional as F

"""
    If the function has in name '_creator' part then it will be invoked during init_objectives_creator.
    Because of that inside it you can create Counter object. Inside other functions do not create counter
    because during target changing the Counter object will be created again, effectively overriting existing data inside logs.
"""

@wrap_objective()
def inner_obj_latent_channel(layer, batch=None):
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[:, ].mean() #TODO - czy to jest prawidłowe, gdy n_channel nie jest używane? Chcemy wszystkie "punkty"
    return inner

@wrap_objective()
def inner_obj_latent(target_layer, target_val, batch=None, loss_f=None):
    inner_loss_f = torch.nn.MSELoss() if loss_f is None else loss_f

    @handle_batch(batch)
    def inner(model):
        latent = model(target_layer)
        latent_target = target_val.repeat(len(latent), 1)
        loss = inner_loss_f(latent, latent_target)
        return loss
    return inner

@wrap_objective()
def inner_obj_latent_obj_max_val_channel(target_layer, target_val: torch.Tensor, batch=None, loss_f=None):
    loss_f = torch.nn.MSELoss() if loss_f is None else loss_f

    @handle_batch(batch)
    def inner(model):
        latent = model(target_layer)
        latent_target = target_val.repeat(len(latent), 1)
        latent = latent[:, 0]
        latent_target = latent_target[:, 0]
        loss = loss_f(latent, latent_target)
        return loss
    return inner

@wrap_objective()
def objective_crossentropy(target_layer, target_val: torch.Tensor, batch=None, loss_f=None):
    loss_f = torch.nn.CrossEntropyLoss()
    @handle_batch(batch)
    def inner(model):
        latent = model(target_layer)
        target_vector = target_val.long().repeat(len(latent))
        loss = loss_f(latent, target_vector)
        return loss

    return inner

@wrap_objective()
def multitarget_channel(layer, n_channel, batch=None):
    """Visualize a single channel"""
    @handle_batch(batch)
    def inner(model):
        out = model(layer)
        return - out[torch.arange(out.size(0)), n_channel].mean()
    return inner

@wrap_objective()
def multitarget_crossentropy(layer, n_channel, batch=None):
    """Visualize a single channel"""
    optimizer = torch.nn.CrossEntropyLoss()
    @handle_batch(batch)
    def inner(model):
        out = model(layer)
        return optimizer(out, n_channel.to(out.device))
    return inner

# =====================================================

def dream_objective_SAE_standalone_diversity(model, **kwargs):
    return - objectives.diversity(
        "model_conv2"
    )

def dream_objective_SAE_diversity(model, **kwargs):
    return - 1e2 * objectives.diversity(model.get_root_objective_target() + 'conv_enc2')

def dream_objective_DLA_diversity_1(model, **kwargs):
    return - 1e2 * objectives.diversity(model.get_root_objective_target() + 'layer4_left_node_conv2')

def dream_objective_DLA_diversity_2(model, **kwargs):
    return - 1e2 * objectives.diversity(model.get_root_objective_target() + 'layer5_prev_root_conv2')

def dream_objective_DLA_diversity_3(model, **kwargs):
    return - 1e2 * objectives.diversity(model.get_root_objective_target() + 'layer6_right_node_conv2')

def dream_objective_resnet18_diversity_1(model, **kwargs):
    return - 1e2 * objectives.diversity(model.get_root_objective_target() + 'layer1_1_conv2')

def dream_objective_resnet18_diversity_2(model, **kwargs):
    return - 1e2 * objectives.diversity(model.get_root_objective_target() + 'layer4_0_conv2')

def dream_objective_latent_channel(model, **kwargs):
    return inner_obj_latent_channel(model.get_objective_target_name())

def dream_objective_channel(target_point: torch.Tensor, model, **kwargs):
    # be careful for recursion by calling methods from source_dataset_obj
    # specify layers names from the model - <top_var_name>_<inner_layer_name>
    # and apply the objective on this layer. Used only for dreams.
    # channel - diversity penalty
    return objectives.channel(model.get_objective_target_name(), target_point.long()) 
    
    #- objectives.diversity(
    #    "model_model_conv2"
    #)

def dream_objective_RESNET20_C100_diversity(model, **kwargs):
    return - 4 * objectives.diversity(model.get_root_objective_target() + "features_final_pool")

def dream_objective_RESNET20_C100_channel(target_point, model, **kwargs):
    return objectives.channel(model.get_objective_target_name(), target_point.long())

def dream_objective_multitarget(target_point:list, model, **kwargs):
    return multitarget_channel(model.get_objective_target_name(), target_point)

def dream_objective_multitarget_crossentropy(target_point:list, model, **kwargs):
    return multitarget_crossentropy(model.get_objective_target_name(), target_point)

def dream_objective_latent_lossf_creator(loss_f=None, **kwargs):
    def inner(target_point, model, **inner_kwargs):
        return inner_obj_latent(
            target_layer=model.get_objective_target_name(), 
            target_val=target_point.to(model.device), 
            loss_f=loss_f,
        ) #- objectives.diversity(
        #    "model_model_conv2"
        #)
    return inner

def dream_objective_latent_step_sample_normal_creator(loss_f, latent_saver: list, std_scale=0.2):
    '''
        latent_saver - must be a list, where the last tensor point 
            from model output layer will be saved at position 0 (zero).
    '''
    def wrapper(target, target_point, model, **kwargs):
        @wrap_objective()
        def dream_objective_step_sample_normal(target, target_layer, target_val, batch=None):
            @handle_batch(batch)
            def inner(model_layers, **kwargs):
                latent = model_layers(target_layer) # return feature map
                out_target_val = torch.normal(target_val, torch.ones_like(target_val) * std_scale)
                latent_saver.append(latent.detach().cpu().numpy())
                latent_target = out_target_val.repeat(len(latent), 1).to(latent.device)
                loss = loss_f(latent, latent_target)
                return loss
            return inner
        return dream_objective_step_sample_normal(
            target_layer=model.get_objective_target_name(), 
            target_val=target_point.to(model.device),  
            target=target
        )
    return wrapper

def dream_objective_latent_lossf_compare_creator(loss_f, latent_saver: list):
    '''
        latent_saver - must be a list, where the last tensor point 
            from model output layer will be saved at position 0 (zero).
    '''
    def wrapper(target, target_point, model, **kwargs):
        @wrap_objective()
        def dream_objective_lossf_latent_compare(target, target_layer, target_val, batch=None):
            @handle_batch(batch)
            def inner(model_layers, **kwargs):
                latent = model_layers(target_layer) # return feature map
                latent_saver.append(latent.detach().cpu().numpy())
                latent_target = target_val.repeat(len(latent), 1)
                loss = loss_f(latent, latent_target)
                return loss
            return inner
        return dream_objective_lossf_latent_compare(
            target_layer=model.get_objective_target_name(), 
            target_val=target_point.to(model.device),  
            target=target
        )
    return wrapper

def dream_objective_latent_neuron_direction(target_point, model, **kwargs):
    return objectives.direction_neuron(model.get_objective_target_name(), target_point.to(model.device))

def test(target, model, source_dataset_obj):
    return objectives.channel(model.get_objective_target_name(), target.long()) - objectives.diversity(
        "vgg_features_10"
    )

@wrap_objective()
def diversity(layer):
    """Encourage diversity between each batch element.

    A neural net feature often responds to multiple things, but naive feature
    visualization often only shows us one. If you optimize a batch of images,
    this objective will encourage them all to be different.

    In particular, it calculates the correlation matrix of activations at layer
    for each image, and then penalizes cosine similarity between them. This is
    very similar to ideas in style transfer, except we're *penalizing* style
    similarity instead of encouraging it.

    Args:
        layer: layer to evaluate activation correlations on.

    Returns:
        Objective.
    """
    cos_sim = torch.nn.CosineSimilarity(dim=1)
    def inner(model):
        layer_t = model(layer)
        batch, channels, _, _ = layer_t.shape
        flattened = layer_t.view(batch, channels, -1)
        grams = torch.matmul(flattened, torch.transpose(flattened, 1, 2))
        grams = F.normalize(grams, p=2, dim=(1, 2))
        sim = cos_sim(grams.view(batch, -1), grams.view(batch, -1))
        flat_batch = grams.view(batch, -1)

        flat_batch_T = flat_batch.T
        z_norm = torch.linalg.norm(flat_batch, dim=1, keepdim=True)
        b_norm = torch.linalg.norm(flat_batch_T, dim=0, keepdim=True)
        cosine_similarity = ((flat_batch @ flat_batch_T) / (z_norm @ b_norm)).T
        cosine_similarity = - cosine_similarity.flatten()[1:].view(batch-1, batch+1)[:,:-1].sum() / batch

        ## too slow !!!
        #b = -sum([ sum([ (grams[i]*grams[j]).sum()
        #       for j in range(batch) if j != i])
        #        for i in range(batch)]) / batch
        return cosine_similarity
    return inner

def dream_objective_SAE_diversity_cosine(model, **kwargs):
    return 0.3 * diversity(model.get_root_objective_target() + 'conv_enc2')

def dream_objective_crossentropy(target_point, model, **kwargs):
    return objective_crossentropy(
        target_layer=model.get_objective_target_name(), 
        target_val=target_point.to(model.device), 
    )

class DreamObjectiveManager():
    GET_OBJECTIVE = {
        'OBJECTIVE-MULTITARGET': dream_objective_multitarget,
        'OBJECTIVE-MULTITARGET-CROSSENTROPY': dream_objective_multitarget_crossentropy,
        'OBJECTIVE-CROSSENTROPY': dream_objective_crossentropy,
        'OBJECTIVE-LATENT-CHANNEL': dream_objective_latent_channel,
        'OBJECTIVE-CHANNEL': dream_objective_channel,
        'OBJECTIVE-RESNET20-C100-CHANNEL': dream_objective_RESNET20_C100_channel,
        'OBJECTIVE-LATENT-LOSSF-CREATOR': dream_objective_latent_lossf_creator,
        'OBJECTIVE-LATENT-NEURON-DIRECTION': dream_objective_latent_neuron_direction,
        'OBJECTIVE-SAE-STANDALONE-DIVERSITY': dream_objective_SAE_standalone_diversity,
        'OBJECTIVE-SAE-DIVERSITY': dream_objective_SAE_diversity_cosine,
        'OBJECTIVE-RESNET20-C100-DIVERSITY': dream_objective_RESNET20_C100_diversity,
        'OBJECTIVE-LATENT-STEP-SAMPLE-NORMAL-CREATOR': dream_objective_latent_step_sample_normal_creator,
        'OBJECTIVE-LATENT-LOSSF-COMPARE-CREATOR': dream_objective_latent_lossf_compare_creator,
        'OBJECTIVE-DLA-DIVERSITY-1': dream_objective_DLA_diversity_1,
        'OBJECTIVE-DLA-DIVERSITY-2': dream_objective_DLA_diversity_2,
        'OBJECTIVE-DLA-DIVERSITY-3': dream_objective_DLA_diversity_3,
        'OBJECTIVE-RESNET-DIVERSITY-1': dream_objective_resnet18_diversity_1,
        'OBJECTIVE-RESNET-DIVERSITY-2': dream_objective_resnet18_diversity_2,
    }
    def __init__(self, dtype: list[str]) -> None:
        if isinstance(dtype, str):
            dtype = [dtype]
        self.dtype = dtype

        self.objectives_f = []
        self.function_names = []

    def __call__(self, *args, **kwargs):
        # calls objective creator and then combines objectives that have @wrap_objective decorator
        combined_objectives = self.first_objective_f(*args, **kwargs)
        for fun in self.objectives_f:
            combined_objectives += fun(*args, **kwargs)
        return combined_objectives

    def get_names(self) -> list[str]:
        return self.function_names

    def init_objectives_creator(self, **kwargs):
        for i in self.dtype:
            i = i.upper()
            obj = DreamObjectiveManager.GET_OBJECTIVE[i]
            if('_creator' in obj.__name__):
                obj = obj(**kwargs)
            self.objectives_f.append(obj)
            self.function_names.append(i)

        self.first_objective_f = self.objectives_f[0]
        self.objectives_f = self.objectives_f[1:]

    def is_latent(self) -> bool:
        b = '_latent' in self.first_objective_f.__name__
        for i in self.objectives_f:
            b = b or '_latent' in i.__name__
        return b