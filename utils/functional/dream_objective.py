import torch
from lucent.optvis.objectives import wrap_objective, handle_batch
from lucent.optvis import objectives
import wandb

counter = 0

@wrap_objective()
def multidim_objective_channel(layer, batch=None):
    @handle_batch(batch)
    def inner(model):
        return -model(layer)[:, ].mean() #TODO - czy to jest prawidłowe, gdy n_channel nie jest używane? Chcemy wszystkie "punkty"
    return inner

@wrap_objective()
def latent_objective_channel(target, target_layer, target_val, logger, batch=None):
    loss_f = torch.nn.MSELoss() 
    global counter
    counter = 0
    @handle_batch(batch)
    def inner(model):
        global counter
        latent = model(target_layer)
        latent_target = target_val.repeat(len(latent), 1)
        loss = loss_f(latent, latent_target)
        logger.log_metrics({f'dream/loss_target_{target}': loss}, counter)
        counter += 1
        return loss
    return inner

def SAE_standalone_multidim_dream_objective_f(model, **kwargs):
    return multidim_objective_channel(model.get_objective_target()) - objectives.diversity(
        "model_conv2"
    )

def SAE_multidim_dream_objective_f( model, **kwargs):
    return multidim_objective_channel(model.get_objective_target()) - objectives.diversity(
        "model_model_conv2"
    )

def SAE_dream_objective_f(target_point, model, **kwargs):
    # be careful for recursion by calling methods from source_dataset_obj
    # specify layers names from the model - <top_var_name>_<inner_layer_name>
    # and apply the objective on this layer. Used only for dreams.
    # channel - diversity penalty
    return objectives.channel(model.get_objective_target(), target_point) - objectives.diversity(
        "model_model_conv2"
    )

def SAE_island_dream_objective_f_creator(logger):
    def SAE_island_dream_objective_f(target, target_point, model, **kwargs):
        return latent_objective_channel(
            target_layer=model.get_objective_target(), 
            target_val=target_point.to(model.device), 
            logger=logger, 
            target=target
        ) - objectives.diversity(
            "model_model_conv2"
        )
    return SAE_island_dream_objective_f

def SAE_island_dream_objective_direction_f(target, target_point, model, source_dataset_obj):
    return objectives.direction_neuron(model.get_objective_target(), target_point.to(model.device)) - objectives.diversity(
        "model_model_conv2"
    )

def default_dream_objective_f(target, model, source_dataset_obj):
    return objectives.direction(model.get_objective_target(), target)

def test(target, model, source_dataset_obj):
    return objectives.channel(model.get_objective_target(), target) - objectives.diversity(
        "vgg_features_10"
    )