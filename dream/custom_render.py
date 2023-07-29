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
from utils import pretty_print as pp
from utils.utils import parse_image_size
import wandb
from utils.utils import hook_model
from dream.image import _Image
from torchvision import transforms as tr
from torch import autocast
from collections.abc import Sequence

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

    def _hook_fun(self, layer, name, tree_name, full_name):
        hook = ModuleHookData()
        self.layers[f'{tree_name}.{name}'] = hook
        def inner(module, input, output):
            hook.hook_f(module=module, input=input, output=output)

        return layer.register_forward_hook(inner)

    def _hook_model(self, model):
        handles = hook_model(model=model, fun=self._hook_fun, hook_to=True)
        return handles

    def unhook(self):
        if(not self.deleted and self.handles is not None):
            for h in self.handles.values():
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
        enable_transforms=True,
        thresholds=None,
        device=None,
        input_image_train_after_hook:list|None=None,
        scheduler=None,
        optim_vals=None,
        autocast_enable=False,
    ) -> None:
        self.preprocess = preprocess
        self.standard_image_size = standard_image_size
        self.display_additional_info = display_additional_info
        self.enable_transforms = enable_transforms
        self.autocast_enable = autocast_enable
        self._set_thresholds(thresholds)
        self._set_custom_f_steps(custom_f_steps)
        self.input_image_train_after_hook = input_image_train_after_hook
        
        self._set_forward_loss_hook(custom_loss_gather_f)
        self._set_advance_end_hook(custom_f)
        self._set_objective_f(objective_f)

        self._set_device(device)
        self._set_optim_image(optim_image)
        self.optim_image.reinit()
        self.optim_vals = optim_vals
        self.optimizer = optimizer
        self._set_model(model)
        self.scheduler = scheduler

        if(transforms is None):
            self._initval_transform_f = None
        elif(not isinstance(transforms, list)):
            self._initval_transform_f = [transforms]
        else:
            self._initval_transform_f = transforms.copy()
        self._set_transform_f(transforms)
        self._set_hook(hook)

    def unhook(self):
        self.hook.unhook()

    def reinit_optim_image(self):
        self.optim_image.reinit()
        self._set_optim_vals(self._initval_optim_val)
        self._set_optimizer(self._initval_optimizer)

    @property
    def model(self) -> torch.nn.Module:
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
            # remove digit from cuda if exist
            if 'cuda' in str(self._device):
                self._device = 'cuda'
            else:
                self._device = 'cpu'
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
            if(isinstance(value, Sequence)):
                self._thresholds = value
            else:
                self._thresholds = (value, )
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
    def scheduler(self):
        return self._scheduler
    @scheduler.setter
    def scheduler(self, value):
        if(not hasattr(self, 'optimizer') or self.optimizer is None):
            raise Exception('Setting scheduler while optimizer is None')
        if(value is None):
            self._scheduler = None
            return
        self._scheduler = value(self.optimizer)


    @property
    def optim_vals(self):
        return self._optim_vals
    @optim_vals.setter
    def optim_vals(self, value):
        self._initval_optim_val = value
        self._set_optim_vals(value)
    def _set_optim_vals(self, value):
        if(value is None):
            value = []
        if(not isinstance(value, list)):
            raise Exception("Optimizer values are not in list!")
        self._optim_vals = self.optim_image.param() + value


    @property
    def optimizer(self) -> torch.optim.Optimizer:
        return self._optimizer
    @optimizer.setter
    def optimizer(self, value):
        self._initval_optimizer = value
        self._set_optimizer(value=value)
        pp.sprint(f"{pp.COLOR.NORMAL}VIS: Optimizer set during visualization config: {pp.COLOR.NORMAL_3}{self._optimizer}")
    def _set_optimizer(self, value):
        if(value is None):
            self._optimizer = torch.optim.Adam(self._optim_vals, lr=1e-3)
        else:
            self._optimizer = value(self._optim_vals)


    @property
    def forward_loss_hook(self):
        return self._forward_loss_hook
    @forward_loss_hook.setter
    def forward_loss_hook(self, value):
        self._set_forward_loss_hook(value=value)
    def _set_forward_loss_hook(self, value):
        self._forward_loss_hook = empty_loss_f if value is None else value

    @property
    def input_image_train_after_hook(self):
        return self._input_image_train_after_hooks
    @input_image_train_after_hook.setter
    def input_image_train_after_hook(self, value:list):
        def inner(image):
            for h in value:
                image = h(image)
            return image
        self._input_image_train_after_hooks = inner if value is not None else lambda x: x

    @property
    def advance_end_hook(self):
        return self._advance_end_hook
    @advance_end_hook.setter
    def advance_end_hook(self, value):
        self._set_objective_f(value=value)
    def _set_advance_end_hook(self, value):
        self._advance_end_hook = empty_f if value is None else value


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
        self._initval_transform_f = value.copy() if isinstance(value, list) else [value]
        self._set_transform_f(value=value)
    def _set_transform_f(self, value):
        if value is None:
            value = transform.standard_transforms
        if(not isinstance(value, list)):
            value = [value]
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
            if(self.display_additional_info and self.enable_transforms):
                pp.sprint(f"{pp.COLOR.WARNING}VIS: Image size before (up/down)sample - {tmp(self.optim_image.image()).shape}")
            value.append(
                torch.nn.Upsample(size=(w, h), mode="bilinear", align_corners=True)
            )

        if(self.enable_transforms):
            if(self.display_additional_info):
                pp.sprint(f"{pp.COLOR.NORMAL_2}VIS: ENABLE DREAM TRANSFORMS")
            self._transform_f = transform.compose(value)
        else:
            if(self.display_additional_info):
                pp.sprint(f"{pp.COLOR.WARNING}VIS: DISABLE ANY DREAM TRANSFORMS")
            self._transform_f = lambda x: x

def normal_step(rd):
    rd.optimizer.zero_grad()
    image = rd.transform_f(rd.optim_image.image())

    with autocast(device_type=rd.device, dtype=torch.float16, enabled=rd.autocast_enable):
        rd.model(image)
        rd.input_image_train_after_hook(image)

        loss = rd.objective_f(rd.hook.get_output)
        loss = rd.forward_loss_hook(loss)
        loss.backward()
        rd.optimizer.step()

def render_vis_loop(
        rd, 
        step_f,
        iterations, 
        verbose=False,
        show_inline=False,
        progress_bar=None,
        refresh_fequency=5,
        return_tensor=True,
    ):
    images = []

    for i in range(1, iterations + 1):
        step_f(rd=rd)
        if i in rd.custom_f_steps:
            rd.advance_end_hook()
        if(rd.scheduler is not None):
            rd.scheduler.step()
        if i in rd.thresholds:
            if(return_tensor):
                image = rd.optim_image.image().detach()
            else:
                image = tensor_to_img_array(rd.optim_image.image())
            if verbose:
                pp.sprint(f"{pp.COLOR.NORMAL_2}Loss at step {i}: {rd.objective_f(rd.hook.get_output):.3f}")
                if show_inline:
                    show_image = tensor_to_img_array(rd.optim_image.image())
                    show(show_image)
            images.append(image)

        if(progress_bar is not None):
            progress_bar.update('vis_iter')
            if i % refresh_fequency == 0:
                progress_bar.refresh()
    return images

def render_vis(
    render_dataclass,
    step_f=None,
    verbose=False,
    show_image=False,
    save_image=False,
    image_name=None,
    show_inline=False,
    progress_bar=None,
    refresh_fequency=5,
    return_tensor=True,
):
    """
        standard_image_size - what image size should be after applying transforms. Upscale / downscale to desired image size.
    """
    rd:RenderVisState = render_dataclass
    rd.reinit_optim_image()

    if verbose:
        rd.model(rd.transform_f(rd.optim_image.image()))
        pp.sprint(f"{pp.COLOR.NORMAL_2}Initial loss: {rd.objective_f(rd.hook.get_output):.3f}")    
    
    iterations = max(rd.thresholds)
    if(progress_bar is not None):
        progress_bar.setup_progress_bar('vis_iter', "[bright_red]Iteration:\n", iterations=iterations)

    model_mode = rd.model.training # from torch.nn.Module
    if model_mode:
        rd.model.eval()

    if(step_f is None):
        step_f = normal_step

    images = render_vis_loop(
        rd=rd,
        step_f=step_f,
        iterations=iterations,
        verbose=verbose,
        show_inline=show_inline,
        progress_bar=progress_bar,
        refresh_fequency=refresh_fequency,
        return_tensor=return_tensor,
    )
        
    if save_image:
        export(rd.optim_image.image(), image_name)
    if show_inline:
        show(tensor_to_img_array(rd.optim_image.image()))
    elif show_image:
        view(rd.optim_image.image())

    if model_mode:
        rd.model.train()
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
