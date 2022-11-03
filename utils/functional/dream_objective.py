import torch
from lucent.optvis.objectives import wrap_objective, handle_batch
from lucent.optvis import objectives
import wandb

counter = {}
for i in range(10):
    counter[i] = 0

@wrap_objective()
def inner_obj_latent_channel(layer, batch=None):
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[:, ].mean() #TODO - czy to jest prawidłowe, gdy n_channel nie jest używane? Chcemy wszystkie "punkty"
    return inner

@wrap_objective()
def inner_obj_latent(target, target_layer, target_val, logger=None, batch=None, loss_f=None):
    inner_loss_f = torch.nn.MSELoss() if loss_f is None else loss_f
    @handle_batch(batch)
    def inner(model):
        global counter
        latent = model(target_layer)
        latent_target = target_val.repeat(len(latent), 1)
        loss = inner_loss_f(latent, latent_target)
        if(logger is not None):
            logger.log_metrics({f'dream/loss_target_{target}': loss}, counter[target])
            #logger.log_metrics({f'dream/test/test_latent_{target}': latent[0, 0]}, counter[target])
            #logger.log_metrics({f'dream/test/test_target_{target}': latent_target[0, 0]}, counter[target])
        counter[target] += 1
        return loss
    return inner

@wrap_objective()
def inner_obj_latent_obj_max_val_channel(target, target_layer, target_val, logger, batch=None):
    loss_f = torch.nn.MSELoss() 
    @handle_batch(batch)
    def inner(model):
        global counter
        latent = model(target_layer)
        latent_target = target_val.repeat(len(latent), 1)
        latent = latent[:, 0]
        latent_target = latent_target[:, 0]
        loss = loss_f(latent, latent_target)
        logger.log_metrics({f'dream/loss_target_{target}': loss}, counter[target])
        #logger.log_metrics({f'dream/test/test_latent_{target}': latent[0]}, counter[target])
        #logger.log_metrics({f'dream/test/test_target_{target}': latent_target[0]}, counter[target])
        counter[target] += 1
        return loss
    return inner

# =====================================================

def dream_objective_SAE_standalone_diversity(model, **kwargs):
    return - objectives.diversity(
        "model_conv2"
    )

def dream_objective_SAE_diversity(model, **kwargs):
    return - objectives.diversity(
        "model_model_conv2"
    )

def dream_objective_latent_channel(model, **kwargs):
    return inner_obj_latent_channel(model.get_objective_target())

def dream_objective_channel(target_point, model, **kwargs):
    # be careful for recursion by calling methods from source_dataset_obj
    # specify layers names from the model - <top_var_name>_<inner_layer_name>
    # and apply the objective on this layer. Used only for dreams.
    # channel - diversity penalty
    return objectives.channel(model.get_objective_target(), target_point) 
    
    #- objectives.diversity(
    #    "model_model_conv2"
    #)

def dream_objective_RESNET20_C100_diversity(model, **kwargs):
    return - 4 * objectives.diversity(model.get_root_objective_target() + "features_final_pool")

def dream_objective_RESNET20_C100_channel(target_point, model, **kwargs):
    return objectives.channel(model.get_objective_target(), target_point)

def dream_objective_latent_lossf_creator(logger=None, loss_f=None, **kwargs):
    def wrapper(target, target_point, model, **inner_kwargs):
        return inner_obj_latent(
            target_layer=model.get_objective_target(), 
            target_val=target_point.to(model.device), 
            logger=logger, 
            target=target,
            loss_f=loss_f,
        ) #- objectives.diversity(
        #    "model_model_conv2"
        #)
    return wrapper

def dream_objective_latent_neuron_direction(target_point, model, **kwargs):
    return objectives.direction_neuron(model.get_objective_target(), target_point.to(model.device))

def test(target, model, source_dataset_obj):
    return objectives.channel(model.get_objective_target(), target) - objectives.diversity(
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
    }
    def __init__(self, dtype: list[str], **kwargs) -> None:
        if isinstance(dtype, str):
            dtype = [dtype]

        self.objectives_f = []
        self.function_names = []

        for i in dtype:
            i = i.upper()
            obj = DreamObjectiveManager.GET_OBJECTIVE[i]
            if('creator' in obj.__name__):
                obj = obj(**kwargs)
            self.objectives_f.append(obj)
            self.function_names.append(i)

        self.first_objectives_f = self.objectives_f[0]
        self.objectives_f = self.objectives_f[1:]

    def __call__(self, *args, **kwargs):
        loss = self.first_objectives_f(*args, **kwargs)
        for fun in self.objectives_f:
            loss += fun(*args, **kwargs)
        return loss

    def get_names(self) -> list[str]:
        return self.function_names