import torch
from lucent.optvis.objectives import wrap_objective, handle_batch
from lucent.optvis import objectives
from utils.counter import Counter, CounterBase, CounterKeys, CounterKeysBase

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
def inner_obj_latent(target, target_layer, target_val, counter: CounterKeysBase, logger=None, label=None, batch=None, loss_f=None):
    inner_loss_f = torch.nn.MSELoss() if loss_f is None else loss_f

    @handle_batch(batch)
    def inner(model):
        latent = model(target_layer)
        latent_target = target_val.repeat(len(latent), 1)
        loss = inner_loss_f(latent, latent_target)
        if(logger is not None and label is not None):
            logger.log_metrics({f'{label}/loss_target_{target}': loss}, counter.get(target))
            #logger.log_metrics({f'dream/sum_latent_{target}': latent.detach().sum().item()}, counter.get())
            #logger.log_metrics({f'dream/sum_latent_target_{target}': latent_target.detach().sum().item()}, counter.get())
            #logger.log_metrics({f'dream/test/test_latent_{target}': latent[0, 0]}, counter.get())
            #logger.log_metrics({f'dream/test/test_target_{target}': latent_target[0, 0]}, counter.get())
        counter.up(target)
        return loss
    return inner

@wrap_objective()
def inner_obj_latent_obj_max_val_channel(target, target_layer, target_val, counter: CounterBase, logger=None, label=None, batch=None, loss_f=None):
    loss_f = torch.nn.MSELoss() if loss_f is None else loss_f

    @handle_batch(batch)
    def inner(model):
        latent = model(target_layer)
        latent_target = target_val.repeat(len(latent), 1)
        latent = latent[:, 0]
        latent_target = latent_target[:, 0]
        loss = loss_f(latent, latent_target)
        if(logger is not None and label is not None):
            logger.log_metrics({f'{label}/loss_target_{target}': loss}, counter.get())
            #logger.log_metrics({f'dream/test/test_latent_{target}': latent[0]}, counter.get())
            #logger.log_metrics({f'dream/test/test_target_{target}': latent_target[0]}, counter.get())
        counter.up()
        return loss
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
    return inner_obj_latent_channel(model.get_objective_target())

def dream_objective_channel(target_point: torch.Tensor, model, **kwargs):
    # be careful for recursion by calling methods from source_dataset_obj
    # specify layers names from the model - <top_var_name>_<inner_layer_name>
    # and apply the objective on this layer. Used only for dreams.
    # channel - diversity penalty
    return objectives.channel(model.get_objective_target(), target_point.long()) 
    
    #- objectives.diversity(
    #    "model_model_conv2"
    #)

def dream_objective_RESNET20_C100_diversity(model, **kwargs):
    return - 4 * objectives.diversity(model.get_root_objective_target() + "features_final_pool")

def dream_objective_RESNET20_C100_channel(target_point, model, **kwargs):
    return objectives.channel(model.get_objective_target(), target_point.long())

def dream_objective_latent_lossf_creator(logger, label=None, loss_f=None, **kwargs):
    counter = CounterKeys()
    def inner(target, target_point, model, **inner_kwargs):
        return inner_obj_latent(
            target_layer=model.get_objective_target(), 
            target_val=target_point.to(model.device), 
            logger=logger, 
            target=target,
            loss_f=loss_f,
            label=label,
            counter=counter,
        ) #- objectives.diversity(
        #    "model_model_conv2"
        #)
    return inner

def dream_objective_latent_step_sample_normal_creator(loss_f, latent_saver: list, std_scale=0.2, logger=None, label=None):
    '''
        latent_saver - must be a list, where the last tensor point 
            from model output layer will be saved at position 0 (zero).
    '''
    counter = Counter()
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
                if(logger is not None and label is not None):
                    logger.log_metrics({f'{label}/step_sample_normal_loss': loss}, counter.get())
                    counter.up()
                return loss
            return inner
        return dream_objective_step_sample_normal(
            target_layer=model.get_objective_target(), 
            target_val=target_point.to(model.device),  
            target=target
        )
    return wrapper

def dream_objective_latent_lossf_compare_creator(loss_f, latent_saver: list, logger=None, label=None):
    '''
        latent_saver - must be a list, where the last tensor point 
            from model output layer will be saved at position 0 (zero).
    '''
    counter = Counter()
    def wrapper(target, target_point, model, **kwargs):
        @wrap_objective()
        def dream_objective_lossf_latent_compare(target, target_layer, target_val, batch=None):
            @handle_batch(batch)
            def inner(model_layers, **kwargs):
                latent = model_layers(target_layer) # return feature map
                latent_saver.append(latent.detach().cpu().numpy())
                latent_target = target_val.repeat(len(latent), 1)
                loss = loss_f(latent, latent_target)
                if(logger is not None and label is not None):
                    logger.log_metrics({f'{label}/lossf_latent_compare_loss': loss}, counter.get())
                    counter.up()
                return loss
            return inner
        return dream_objective_lossf_latent_compare(
            target_layer=model.get_objective_target(), 
            target_val=target_point.to(model.device),  
            target=target
        )
    return wrapper

def dream_objective_latent_neuron_direction(target_point, model, **kwargs):
    return objectives.direction_neuron(model.get_objective_target(), target_point.to(model.device))

def test(target, model, source_dataset_obj):
    return objectives.channel(model.get_objective_target(), target.long()) - objectives.diversity(
        "vgg_features_10"
    )


class DreamObjectiveManager():
    GET_OBJECTIVE = {
        'OBJECTIVE-LATENT-CHANNEL': dream_objective_latent_channel,
        'OBJECTIVE-CHANNEL': dream_objective_channel,
        'OBJECTIVE-RESNET20-C100-CHANNEL': dream_objective_RESNET20_C100_channel,
        'OBJECTIVE-LATENT-LOSSF-CREATOR': dream_objective_latent_lossf_creator,
        'OBJECTIVE-LATENT-NEURON-DIRECTION': dream_objective_latent_neuron_direction,
        'OBJECTIVE-SAE-STANDALONE-DIVERSITY': dream_objective_SAE_standalone_diversity,
        'OBJECTIVE-SAE-DIVERSITY': dream_objective_SAE_diversity,
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