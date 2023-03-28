# Copyright 2020 The Lucent Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import numpy as np
from PIL import Image
import torch

from lucent.optvis import objectives
from dream import transform
from lucent.misc.io import show
from colorama import Fore, Back, Style
from utils.utils import parse_image_size
import wandb
from utils.utils import hook_model
from dream.image import _Image
from torchvision import transforms as tr

dream_current_step = 0

def _check_img_size(transform_f, image_f, standard_image_size):
    if(standard_image_size is None):
        return
    trans_size = list(transform_f(image_f()).shape)
    if(len(trans_size) != len(standard_image_size) + 1):
        raise Exception(f'standard_image_size size {len(standard_image_size)}#{standard_image_size} is not equal \
image size of {len(trans_size)}#{trans_size[1:]} after transforms.')

    new_standard_image_size = [0] + list(standard_image_size) # add dummy batch size
    for idx, (created, standard) in enumerate(zip(trans_size, new_standard_image_size)):
        if idx == 0:
            continue
        if(created != standard):
            raise Exception(f'At {idx} bad image size. Should be {standard}#{standard_image_size} but is {created}#{trans_size[1:]}.')

def empty_f():
    return

def empty_loss_f(loss):
    return loss

class ModuleHookData:
    def __init__(self):
        self.module = None
        self.output = None

    def hook_f(self, module, input, output):
        self.module = module
        self.output = output

class ModelHookData():
    def __init__(self, model) -> None:
        self.layers = dict()
        self.deleted = False

        self.handles = None
        self.handles = self._hook_model(model=model)

    def _hook_fun(self, layer, name, tree_name):
        hook = ModuleHookData()
        self.layers[f'{tree_name}_{name}'] = hook
        def inner(module, input, output):
            hook.hook_f(module=module, input=input, output=output)

        return inner

    def _hook_model(self, model):
        handles = hook_model(model=model, fun=self._hook_fun, hook_to=True)
        return handles

    def unhook(self):
        if(not self.deleted and self.handles is not None):
            for _, _, h in self.handles:
                h.remove()
            self.deleted = True

    def __del__(self):
        self.unhook()

    def get_output(self, layer):
        return self.layers[layer].output

class RenderVisState():
    def __init__(
        self,
        model,
        optim_image: _Image,
        objective_f=None,
        hook: ModelHookData=None,
        optimizer: torch.optim.Optimizer=None,
        custom_loss_gather_f=None,
        custom_f=None,
        custom_f_steps=None,
        transforms=None,
        preprocess=True,
        standard_image_size=None,
        display_additional_info=False,
        disable_transforms=False,
        thresholds=None,
        device=None,
    ) -> None:
        self.preprocess = preprocess
        self.standard_image_size = standard_image_size
        self.display_additional_info = display_additional_info
        self.disable_transforms = disable_transforms
        self._set_thresholds(thresholds)
        self._set_custom_f_steps(custom_f_steps)
        
        self._set_custom_loss_gather_f(custom_loss_gather_f)
        self._set_custom_f(custom_f)
        self._set_objective_f(objective_f)

        self._set_device(device)
        self._set_optim_image(optim_image)
        self._set_optimizer(optimizer)
        self._set_model(model)
        self._set_transform_f(transforms)
        self._set_hook(hook)

    def unhook(self):
        self.hook.unhook()

    @property
    def model(self):
        return self._model
    @model.setter
    def model(self, value):
        self._set_model(value=value)
    def _set_model(self, value):
        self._model = value.to(self.device)
        if(hasattr(self, '_initval_transform_f')):
            self._set_transform_f(self._initval_transform_f)
        if(hasattr(self, '_hook')):
            self._set_hook(None)


    @property
    def device(self) -> str:
        return self._device
    @device.setter
    def device(self, value):
        self._set_device(value=value)
    def _set_device(self, value):
        if(value is None):
            self._device = 'cpu'
        else:
            self._device = value
        if(hasattr(self, '_model')):
            self._model = self._model.to(self._device)
        if(hasattr(self, '_optim_image')):
            self._optim_image = self._optim_image.to(self._device)
        if(hasattr(self, '_optimizer')):
            self._set_optimizer(self._initval_optimizer)


    @property
    def optim_image(self):
        return self._optim_image
    @optim_image.setter
    def optim_image(self, value):
        self._set_optim_image(value=value)
    def _set_optim_image(self, value):
        self._optim_image = value.to(self._device)


    @property
    def hook(self):
        return self._hook
    @hook.setter
    def hook(self, value):
        self._set_hook(value=value)
    def _set_hook(self, value):
        if(value is None):
            self._hook = ModelHookData(model=self.model)
        else:
            self._hook = value

        

    @property
    def thresholds(self) -> tuple|list:
        """
            max() - how many iterations is used to generate an input image. Also
            for the rest of the numbers, save the image during the i-th iteration
        """
        return self._thresholds
    @thresholds.setter
    def thresholds(self, value):
        self._set_thresholds(value=value)
    def _set_thresholds(self, value):
        if(value is not None):
            self._thresholds = value
        else:
            self._thresholds = (512,)


    @property
    def custom_f_steps(self):
        return self._custom_f_steps
    @custom_f_steps.setter
    def custom_f_steps(self, value):
        self._set_custom_f_steps(value=value)
    def _set_custom_f_steps(self, value):
        if(value is not None):
            self._custom_f_steps = value
        else:
            self._custom_f_steps = (0,)


    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._set_optimizer(value=value)
    def _set_optimizer(self, value):
        self._initval_optimizer = value
        if(value is None):
            self._optimizer = torch.optim.Adam(self.optim_image.param(), lr=1e-3)
        else:
            self._optimizer = value(self.optim_image.param())


    @property
    def custom_loss_gather_f(self):
        return self._custom_loss_gather_f
    @custom_loss_gather_f.setter
    def custom_loss_gather_f(self, value):
        self._set_custom_loss_gather_f(value=value)
    def _set_custom_loss_gather_f(self, value):
        self._custom_loss_gather_f = empty_loss_f if value is None else value


    @property
    def custom_f(self):
        return self._custom_f
    @custom_f.setter
    def custom_f(self, value):
        self._set_objective_f(value=value)
    def _set_custom_f(self, value):
        self._custom_f = empty_f if value is None else value


    @property
    def objective_f(self):
        if(self._objective_f is None):
            raise Exception('Objective function not set!')
        return self._objective_f
    @objective_f.setter
    def objective_f(self, value):
        self._set_objective_f(value=value)
    def _set_objective_f(self, value):
        if(value is not None):
            self._objective_f = objectives.as_objective(value)
        else:
            self._objective_f = None
    
    @property
    def transform_f(self):
        return self._transform_f
    @transform_f.setter
    def transform_f(self, value):
        self._set_transform_f(value=value)
    def _set_transform_f(self, value):
        self._initval_transform_f = value.copy()
        if value is None:
            value = transform.standard_transforms
        value = [tr.Lambda(lambda x: x.to(self.device))] + value.copy()
        if self.preprocess:
            if self.model._get_name() == "InceptionV1":
                # Original Tensorflow InceptionV1 takes input range [-117, 138]
                value.append(transform.preprocess_inceptionv1())
            else:
                # Assume we use normalization for torchvision.models
                # See https://pytorch.org/docs/stable/torchvision/models.html
                value.append(transform.normalize())

        if self.standard_image_size is not None:
            new_size = self.standard_image_size
        else:
            new_size = None
        if new_size:
            _, w, h = parse_image_size(new_size)
            tmp = transform.compose(value)
            if(self.display_additional_info):
                print(f"VIS: Image size before (up/down)sample - {tmp(self.optim_image.image()).shape}")
            value.append(
                torch.nn.Upsample(size=(w, h), mode="bilinear", align_corners=True)
            )

        if(self.disable_transforms):
            if(self.display_additional_info):
                print(f"{Fore.RED}VIS: DISABLE ANY DREAM TRANSFORMS{Style.RESET_ALL}")
            self._transform_f = lambda x: x
        else:
            if(self.display_additional_info):
                print("VIS: ENABLE DREAM TRANSFORMS")
            self._transform_f = transform.compose(value)

def render_vis(
    render_dataclass,
    verbose=False,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
    progress_bar=None,
    refresh_fequency=50,
):
    """
        standard_image_size - what image size should be after applying transforms. Upscale / downscale to desired image size.
    """
    rd:RenderVisState = render_dataclass

    if verbose:
        rd.model(rd.transform_f(rd.optim_image.image()))
        print("Initial loss: {:.3f}".format(rd.objective_f(rd.hook.get_output)))
    
    images = []
    iterations = max(rd.thresholds)
    progress_bar.setup_iteration(iterations=iterations)
    for i in range(1, iterations + 1):
        #print(torch.sum(torch.abs(transform_f(image_f()))))
        def closure():
            rd.optimizer.zero_grad()
            rd.model(rd.transform_f(rd.optim_image.image()))
            loss = rd.objective_f(rd.hook.get_output)
            loss = rd.custom_loss_gather_f(loss)
            global dream_current_step
            wandb.log({'loss_during_dreaming': loss.item()}, step=dream_current_step)
            dream_current_step += 1
            loss.backward()
            return loss
            
        rd.optimizer.step(closure)
        if i in rd.custom_f_steps:
            rd.custom_f()
        if i in rd.thresholds:
            image = tensor_to_img_array(rd.optim_image.image())
            if verbose:
                print("Loss at step {}: {:.3f}".format(i, rd.objective_f(rd.hook.get_output)))
                if show_inline:
                    show(image)
            images.append(image)

        progress_bar.update_iteration()
        if i % refresh_fequency == 0:
            progress_bar.refresh()

    if save_image:
        export(rd.optim_image.image(), image_name)
    if show_inline:
        show(tensor_to_img_array(rd.optim_image.image()))
    elif show_image:
        view(rd.optim_image.image())
    #rd.hook.unhook()
    return images

def tensor_to_img_array(tensor):
    image = tensor.cpu().detach().numpy()
    image = np.transpose(image, [0, 2, 3, 1])
    return image

def view(tensor):
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).show()

def export(tensor, image_name=None):
    image_name = image_name or "image.jpg"
    image = tensor_to_img_array(tensor)
    assert len(image.shape) in [
        3,
        4,
    ], "Image should have 3 or 4 dimensions, invalid image shape {}".format(image.shape)
    # Change dtype for PIL.Image
    image = (image * 255).astype(np.uint8)
    if len(image.shape) == 4:
        image = np.concatenate(image, axis=1)
    Image.fromarray(image).save(image_name)


#class ModuleHook:
#    def __init__(self, module):
#        self.hook = module.register_forward_hook(self.hook_fn)
#        self.module = None
#        self.features = None
#
#    def hook_fn(self, module, input, output):
#        self.module = module
#        self.features = output
#
#    def close(self):
#        self.hook.remove()

#def hook_model(model, optim_image):
#    features = OrderedDict()
#
#    # recursive hooking function
#    def hook_layers(net, prefix=[]):
#        if hasattr(net, "_modules"):
#            for name, layer in net._modules.items():
#                if layer is None:
#                    # e.g. GoogLeNet's aux1 and aux2 layers
#                    continue
#                features["_".join(prefix + [name])] = ModuleHook(layer)
#                hook_layers(layer, prefix=prefix + [name])
#
#    hook_layers(model)
#
#    def hook(layer):
#        if layer == "input":
#            out = optim_image.image()
#        elif layer == "labels":
#            out = list(features.values())[-1].features
#        else:
#            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
#            out = features[layer].features
#        assert out is not None, "Hook was not set properly. Model output during forward pass was not saved. May be because of error during forward pass. \
#Check first traceback. \
#There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
#        return out
#
#    return hook
