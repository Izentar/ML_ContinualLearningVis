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

from __future__ import absolute_import, division, print_function

import warnings
from collections import OrderedDict
import numpy as np
from tqdm import tqdm
from PIL import Image
import torch

from lucent.optvis import objectives, transform, param
from lucent.misc.io import show
from colorama import Fore, Back, Style
from utils.utils import parse_image_size

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

def render_vis(
    model,
    objective_f,
    param_f=None,
    optimizer=None,
    transforms=None,
    thresholds=(512,),
    custom_f_steps=(0,),
    custom_f=None,
    custom_loss_gather_f=None,
    verbose=False,
    preprocess=True,
    show_image=True,
    save_image=False,
    image_name=None,
    show_inline=False,
    disable_transforms=False,
    progress_bar=None,
    refresh_fequency=50,
    standard_image_size=None,
    display_additional_info=True,
):
    """
        standard_image_size - what image size should be after applying transforms. Upscale / downscale to desired image size.
    """
    if param_f is None:
        param_f = lambda: param.image(128)
    # param_f is a function that should return two things
    # params - parameters to update, which we pass to the optimizer
    # image_f - a function that returns an image as a tensor
    params, image_f = param_f()

    if(custom_loss_gather_f is None):
        custom_loss_gather_f = empty_loss_f
    
    if(custom_f is None):
        custom_f = empty_f

    if optimizer is None:
        optimizer = lambda params: torch.optim.Adam(params, lr=1e-3)
    optimizer = optimizer(params)

    if transforms is None:
        transforms = transform.standard_transforms
    transforms = transforms.copy()

    if preprocess:
        if model._get_name() == "InceptionV1":
            # Original Tensorflow InceptionV1 takes input range [-117, 138]
            transforms.append(transform.preprocess_inceptionv1())
        else:
            # Assume we use normalization for torchvision.models
            # See https://pytorch.org/docs/stable/torchvision/models.html
            transforms.append(transform.normalize())

    if standard_image_size is not None:
        new_size = standard_image_size
    else:
        new_size = None
    if new_size:
        _, w, h = parse_image_size(new_size)
        transforms.append(
            torch.nn.Upsample(size=(w, h), mode="bilinear", align_corners=True)
        )

    if(disable_transforms):
        if(display_additional_info):
            print(f"{Fore.RED}INFO: DISABLE ANY DREAM TRANSFORMS{Style.RESET_ALL}")
        transform_f = lambda x: x
    else:
        if(display_additional_info):
            print("INFO: ENABLE DREAM TRANSFORMS")
        transform_f = transform.compose(transforms)

    model.eval()
    hook = hook_model(model, image_f)
    objective_f = objectives.as_objective(objective_f)

    if verbose:
        model(transform_f(image_f()))
        print("Initial loss: {:.3f}".format(objective_f(hook)))
    
    images = []
    iterations = max(thresholds)
    progress_bar.setup_iteration(iterations=iterations)
    for i in range(1, iterations + 1):
        def closure():
            optimizer.zero_grad()
            model(transform_f(image_f()))
            loss = objective_f(hook)
            loss = custom_loss_gather_f(loss)
            loss.backward()
            return loss
            
        optimizer.step(closure)
        if i in custom_f_steps:
            custom_f()
        if i in thresholds:
            image = tensor_to_img_array(image_f())
            if verbose:
                print("Loss at step {}: {:.3f}".format(i, objective_f(hook)))
                if show_inline:
                    show(image)
            images.append(image)

        progress_bar.update_iteration()
        if i % refresh_fequency == 0:
            progress_bar.refresh()

    if save_image:
        export(image_f(), image_name)
    if show_inline:
        show(tensor_to_img_array(image_f()))
    elif show_image:
        view(image_f())
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


class ModuleHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)
        self.module = None
        self.features = None

    def hook_fn(self, module, input, output):
        self.module = module
        self.features = output

    def close(self):
        self.hook.remove()


def hook_model(model, image_f):
    features = OrderedDict()

    # recursive hooking function
    def hook_layers(net, prefix=[]):
        if hasattr(net, "_modules"):
            for name, layer in net._modules.items():
                if layer is None:
                    # e.g. GoogLeNet's aux1 and aux2 layers
                    continue
                features["_".join(prefix + [name])] = ModuleHook(layer)
                hook_layers(layer, prefix=prefix + [name])

    hook_layers(model)

    def hook(layer):
        if layer == "input":
            out = image_f()
        elif layer == "labels":
            out = list(features.values())[-1].features
        else:
            assert layer in features, f"Invalid layer {layer}. Retrieve the list of layers with `lucent.modelzoo.util.get_model_layers(model)`."
            out = features[layer].features
        assert out is not None, "Hook was not set properly. Model output during forward pass was not saved. May be because of error during forward pass. \
Check first traceback. \
There are no saved feature maps. Make sure to put the model in eval mode, like so: `model.to(device).eval()`. See README for example."
        return out

    return hook
