from typing import Any
import torch
from dream.custom_render import render_vis, RenderVisState
from dream.image import Image
from utils.functional import dream_objective as do
import wandb
import numpy as np

class SingleDream():
    def __init__(self, drift_sigma, num_classes, logger, loss_f=None, deep_inversion=True) -> None:
        self.drift_sigma = drift_sigma
        self.loss_f = loss_f if loss_f is not None else torch.nn.MSELoss()
        self.num_classes = num_classes
        self.logger = logger

    def _setup_deep_inv(self):
        pass

    def _apply_normal_drift(self, target):
        drift = torch.normal(torch.zeros_like(target), torch.ones_like(target) * self.drift_sigma)
        return target + drift
    
    def _create_wandb_table(self, point, cl):
        point = point.detach().cpu().squeeze()
        point = point.numpy().squeeze()
        columns = [f'd{i}' for i in range(point.shape[0])]
        point = np.append(point, cl)
        table_main = wandb.Table(columns=columns + ['class_type'])
        table_main.add_data(*point)
        return table_main

    def _log_images(self, imgs, orig_target, render_target, cl):
        label = 'single_dream'
        self.logger.log_metrics({f'{label}/constructed_dreams': [
                wandb.Image(
                    i,
                ) for i in imgs
            ]
        })

        wandb.log({f'{label}/original_target': self._create_wandb_table(orig_target, cl)})
        wandb.log({f'{label}/render_target': self._create_wandb_table(render_target, cl)})

    def __call__(self, model, device='cuda', vis_class=1, image_type='pixel', iters=1000, render_transforms=None,
                 standard_image_size=32) -> Any:
        render_point = torch.zeros(self.num_classes, device=device, dtype=torch.float)
        render_point[vis_class] = 1.
        orig_target = render_point.clone()
        render_point = self._apply_normal_drift(render_point)

        objective_f = do.dream_objective_latent_lossf_creator(loss_f=self.loss_f)(target_point=render_point, model=model)
        objective_f = do.dream_objective_channel(target_point=render_point, model=model)



        state = RenderVisState(
            model=model,
            optim_image=Image(image_type, w=32, batch=1),
            objective_f=objective_f,
            optimizer=lambda param: torch.optim.Adam(param, lr=0.003, betas=(0.9, 0.99)),
            thresholds=iters,
            enable_transforms=True,
            transforms=render_transforms,
            device=device,
            standard_image_size=32,
        )
        imgs = render_vis(
            render_dataclass=state,
        )

        self._log_images(imgs=imgs, orig_target=orig_target, render_target=render_point, cl=vis_class)