import torch
from lucent.optvis.objectives import wrap_objective, handle_batch
from lucent.optvis import objectives
import wandb

counter = {}
for i in range(10):
    counter[i] = 0

@wrap_objective()
def inner_obj_multidim_channel(layer, batch=None):
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[:, ].mean() #TODO - czy to jest prawidłowe, gdy n_channel nie jest używane? Chcemy wszystkie "punkty"
    return inner

@wrap_objective()
def inner_obj_latent(target, target_layer, target_val, logger=None, batch=None):
    loss_f = torch.nn.MSELoss() 
    @handle_batch(batch)
    def inner(model):
        global counter
        latent = model(target_layer)
        latent_target = target_val.repeat(len(latent), 1)
        loss = loss_f(latent, latent_target)
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

def dream_objective_SAE_standalone_multidim(model, **kwargs):
    return inner_obj_multidim_channel(model.get_objective_target()) - objectives.diversity(
        "model_conv2"
    )

def dream_objective_SAE_multidim( model, **kwargs):
    return inner_obj_multidim_channel(model.get_objective_target()) - objectives.diversity(
        "model_model_conv2"
    )

def dream_objective_SAE_channel(target_point, model, **kwargs):
    # be careful for recursion by calling methods from source_dataset_obj
    # specify layers names from the model - <top_var_name>_<inner_layer_name>
    # and apply the objective on this layer. Used only for dreams.
    # channel - diversity penalty
    return objectives.channel(model.get_objective_target(), target_point) 
    
    #- objectives.diversity(
    #    "model_model_conv2"
    #)

def dream_objective_RESNET20_C100_pretrined(target_point, model, **kwargs):
    return objectives.channel(model.get_objective_target(), target_point) - 4 * objectives.diversity(model.get_root_objective_target() + "features_final_pool")

def dream_objective_SAE_island_creator(logger):
    def SAE_island_dream_objective_f(target, target_point, model, **kwargs):
        return inner_obj_latent(
            target_layer=model.get_objective_target(), 
            target_val=target_point.to(model.device), 
            logger=logger, 
            target=target
        ) #- objectives.diversity(
        #    "model_model_conv2"
        #)
    return SAE_island_dream_objective_f

def dream_objective_SAE_island_direction(target, target_point, model, source_dataset_obj):
    return objectives.direction_neuron(model.get_objective_target(), target_point.to(model.device)) - objectives.diversity(
        "model_model_conv2"
    )

def dream_objective_default(target, model, source_dataset_obj):
    return objectives.direction(model.get_objective_target(), target)

def test(target, model, source_dataset_obj):
    return objectives.channel(model.get_objective_target(), target) - objectives.diversity(
        "vgg_features_10"
    )